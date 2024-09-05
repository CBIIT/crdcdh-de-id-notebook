import pydicom, io, sys
import pytesseract
import re
import numpy as np
from PIL import Image, ImageDraw
from botocore.exceptions import ClientError
from common.utils import process_dict_tags, get_date_time, dict2yaml, load_json_file, yaml2dict, generate_regex, save_dict_to_json, convert_json_to_csv
from common.de_id_utils import get_pii_boxes
from common.pixel_utils import parse_pixel_data, enhance_image, reverse_windowing
from common.constants import DICOM_UID_MAP_JSON, EMPTY_STRING, PATIENT_ID_MAP_JSON, PATIENT_SEQUENCES_JSON, ANONYMIZED
class ProcessMedImage:
    def __init__(self, boto3_session, rule_config_file_path, silence_mode = False):
        """
        constructor of ProcessMedImage
        """
        self.quiet = silence_mode
        self.boto3_session = boto3_session #get_boto3_session()
        self.timestamp = get_date_time()
        self.data_time = get_date_time("%Y-%m-%d-%H-%M-%S")
        # self.role = get_sagemaker_execute_role(self.boto3_session)
        self.s3_client = self.boto3_session.client('s3') if self.boto3_session else None
        self.rekognition= None
        self.comprehend_medical = None
        # set rules
        self.rule_config_file_path = rule_config_file_path
        self.rules = None
        self.dicom_tags = None
        self.phi_tags = None
        self.sensitive_words = None
        self.vr = None
        self.regex = None
        self.confidence_threshold = None
        self.set_rules(rule_config_file_path)
        # dicom data
        self.local_dicom_path = None
        self.ds = None
        self.pixel_array = None
        self.image_data = None
        self.patient_id = None
        self.studyInstanceUID = None
        self.seriesInstanceUID = None
        #StudyInstanceID amd Series instance UID map
        self.dicom_uid_map = load_json_file(DICOM_UID_MAP_JSON)
        self.sequence = load_json_file(PATIENT_SEQUENCES_JSON)
        self.patient_id_map = load_json_file(PATIENT_ID_MAP_JSON)

    def set_rules(self, rule_config_file_path):
        """
        set rules for DICOM tags and keywords
        """
        # Load the rules from the YAML file
        self.rules = yaml2dict(rule_config_file_path)["rules"]
        self.dicom_tags = self.rules['dicom_tags']
        self.phi_tags = set([ tuple(item["tag"]) for item in self.dicom_tags ])
        # print(self.phi_tags)
        self.sensitive_words = self.rules['keywords']
        # print(self.sensitive_words)
        self.regex = set(self.rules['regex'])
        self.confidence_threshold = int(self.rules['confidence_threshold'])
        self.pii_patterns = re.compile(r'\b(?:{0})\b'.format('|'.join(self.regex)))

    def parse_dicom_file(self, bucket, key, local_dicom_path, useAI = False):
        """
        parse dicom file and upload DICOM file to s3 bucket for de-identification.
        :param src_bucket: source s3 bucket
        :param src_key: source s3 key
        :param local_dicom_path: local dicom file path
        :return: True if the upload is successful, False otherwise
        :rtype: bool, pydicom.dataset.FileDataset
        """

        try:
            self.local_dicom_path = local_dicom_path
            self.image_data = None
            # Load the DICOM data
            self.ds = pydicom.dcmread(local_dicom_path)
            if 'PlanarConfiguration' not in self.ds:
                self.ds.PlanarConfiguration = 0
            # inspect_dicom_file(self.ds)
            # Extract pixel array and convert to uint8 (if necessary)
            np.seterr(divide='ignore', invalid='ignore')
            number_of_frames = self.ds.get('NumberOfFrames', 1) 
            if not self.quiet:
                print(f"number_of_frames: {number_of_frames}")
            try: 
                self.pixel_array = self.ds.pixel_array
                min_pixel = np.min(self.pixel_array) if number_of_frames < 2 else np.min(self.pixel_array[0])
                max_pixel = np.max(self.pixel_array) if number_of_frames < 2 else np.max(self.pixel_array[0])
                if not self.quiet:
                    print(f"min_pixel: {min_pixel}, max_pixel: {max_pixel}")
                self.image_data = parse_pixel_data(self.ds, max_pixel, min_pixel)
            except Exception as e:
                print(f"Error parsing pixel data in {local_dicom_path}: {e}")  
                return False
            if useAI:
                # Convert the pixel array to a PIL Image
                try:
                    image = Image.fromarray(self.image_data)
                    image_io = io.BytesIO()
                    image.save(image_io, format='PNG')
                    image_io.seek(0)
                    self.s3_client.upload_fileobj(image_io, bucket, key)
                except Exception as e:
                    print(f"Error converting pixel data to image in {"/".join(self.local_dicom_path.split("/")[5:])}: {e}")
                    return False
               
            return True
        except ClientError as ce:
            print(f"Error parsing DICOM file {local_dicom_path}: {ce}")
            raise
        except Exception as e:
            print(f"Error parsing DICOM file {local_dicom_path}: {e}")
            raise

    def de_identify_dicom(self):
        """
        de-identify DICOM metadata with HIPPA Privacy Rules
        """
        if not self.quiet:
            print("De-identifying DICOM metadata")
        if not self.quiet:
            print("De-identifying DICOM metadata")
        # Redact PHI in the DICOM dataset
        redacted = 0
        redacted_value = "None"
        detected_tags = []
        for item in self.ds.iterall():
            vr = item.VR
            if item.value in [None, 'None', "", "none"] or vr in ["OW"]: continue 
            name = item.name
            tuple = (item.tag.group, item.tag.element)
            in_keywords = [key for key in self.sensitive_words if key in name]
            if len(in_keywords) > 0:
                redacted_value = None
            else:
                if tuple in self.phi_tags or vr in ["UI", "PN", "DA", "DT", "TM"]:
                    redacted_value = self.redact_tag_value(item.value, tuple, vr)
                elif "DateTime" in name:
                    redacted_value = "00010101010101.000000+0000"
                elif "time stamp" in name:
                    redacted_value = "0000000000"
                else:
                    redacted_value = "None"
            if redacted_value != "None":
                if not self.quiet:
                    print(f"Tag: {item} - Redacted Value: {redacted_value}") 
                item.value = redacted_value
                detected_tags.append(tuple)
                redacted += 1
        if not self.quiet:
            print(f"Redacted DICOM matadata")
        return redacted, detected_tags

    def detect_id_in_tags(self):
        """
        detect PHI info in DICOM by Comprehend Medical
        """
        tags = []
        detected_elements = []
        all_ids = []
        ids_list = []
        ids = []
       
        if not self.comprehend_medical:
            # initial AI foe evaluation
            self.comprehend_medical = self.boto3_session.client('comprehendmedical')
        for item in self.ds.iterall():
            # Detect PHI/PII in the text
            tuple = (item.tag.group, item.tag.element)
            # Escape pixel data, redacted values, pixel data tags
            if item.value in [None, 'None', "", "none", EMPTY_STRING, ANONYMIZED, 0, "0", b"ANONYMIZED"] \
                or item.VR in ["OW","UI", "PN", "DA", "DT", "TM"] or tuple in [(0x7FE0, 0x0010), ((0x6000, 0x3000))]: 
                continue 
            tuple = None
            if isinstance(item.value, str):
                ids = self.detect_id_in_text_AI(item.value, is_image=False)
                if ids and len(ids) > 0:
                    if not self.quiet:
                        print(f"Tag: {item}")
                    if not self.quiet:
                        print(f"Tag: {item}")
                    all_ids.extend(ids)
                    detected_elements.append(item)
                    
                    tags.append(tuple)
                    # self.dicom_tags.append({"tag": tuple, "name": item.name})
            elif isinstance(item.value, list):
                for itm in item.value:
                    if not itm: pass
                    if isinstance(itm, dict):
                        for key, value in itm.items():
                            if not value: pass
                            ids = self.detect_id_in_text_AI(value, is_image=False)
                            ids_list.extend(ids)
                            if ids and len(ids) > 0 and not self.quiet:
                                print(f"Tag: {item}")
                            if ids and len(ids) > 0 and not self.quiet:
                                print(f"Tag: {item}")
                    else:
                        ids = self.detect_id_in_text_AI(item, is_image=False)
                        ids_list.extend(ids)
                        if ids and len(ids) > 0:
                            print(f"Tag: {item}")
                            print(f"Tag: {item}")
                if len(ids_list) > 0:
                    if not self.quiet:
                        print(f"Tag: {item}")
                    detected_elements.append(item)
                    tuple = (item.tag.group, item.tag.element)
                    tags.append(tuple)

            if tuple and tuple not in self.phi_tags:
                self.dicom_tags.append({"tag": tuple, "name": item.name})     
        return detected_elements, tags, all_ids, 
    
    def redact_tags(self, elements):
        """
        redact PHI info in DICOM by Comprehend Medical
        """
        for item in elements:
            value = item.value
            item.value = self.redact_tag_value(value, (item.tag.group, item.tag.element), item.VR)
    
    def detect_id_in_text_AI(self, detected_text, is_image = False):
        """
        detect PHI info in text by Comprehend Medical
        """
        """
        detect PHI info in text by Comprehend Medical
        """
        ids = []
        if not detected_text: return ids
        text = detected_text['DetectedText'] if is_image else detected_text
        phi_entities = self.analyze_text_for_phi(text)
        valid_entities = []
        if phi_entities and len(phi_entities) > 0:
            for entity in phi_entities:
                if entity['Score'] > max(0.85, self.confidence_threshold/100): 
                    valid_entities.append(entity)
                    if not self.quiet:
                        print(f"Entity: {entity['Text']} - Type: {entity['Type']} - Confidence: {entity['Score']}")
                    if is_image and entity['Type'] in ["ID", "AGE", "ADDRESS", "PHONE_OR_FAX", "DATE"]:
                        regex = generate_regex(entity['Text'])
                        if regex and regex not in self.regex:
                            self.regex.add(regex)
            if len(valid_entities) > 0:
                ids.append(text)
        return ids
        
    def detect_id_in_img(self, bucket_name, image_key,  use_AI= False):
        """
        detect phi in image with or without AI
        """
        all_ids = []
        extracted_text = False
        if self.image_data is None:
            return all_ids, extracted_text
        if use_AI:
            # Use Rekognition to detect text
            if not self.rekognition:
                self.rekognition = self.boto3_session.client('rekognition')
            if not self.comprehend_medical:
                self.comprehend_medical = self.boto3_session.client('comprehendmedical')
            # Detect text in the image using Rekognition
            response = self.rekognition.detect_text(Image={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': image_key
                }
            })
            detected_texts = [text for text in response['TextDetections']]
            extracted_text = (len(detected_texts) > 0)
            if not extracted_text: return all_ids, extracted_text
            img_width, img_height = self.ds.Columns, self.ds.Rows  # Number of rows corresponds to the height
            for text in detected_texts:
                # add rules
                if text['DetectedText']  in ["LACH", "SWU"]: continue
                # use Comprehend Medical detect PHI in text
                # print(text)
                ids = self.detect_id_in_text_AI(text, True)
                if ids and len(ids) > 0:
                    box = text['Geometry']['BoundingBox']
                    left = img_width * box['Left']
                    top = img_height * box['Top']
                    width = img_width * box['Width']
                    height = img_height * box['Height']
                    all_ids.append({"Text": text['DetectedText'], "Text Block": {'Left': left, 'Top': top, 'Width': width, 'Height': height}})
        else:
            image = enhance_image(self.image_data, self.local_dicom_path, False)
            if not image: return all_ids, extracted_text
            # expend the image for detecting
            width, height = image.size
            image_exp = image.resize((width * 2, height * 2))
            expanded_width, expanded_height = image_exp.size
            x_scale = width / expanded_width
            y_scale = height / expanded_height
            text_boxes = []
            # Initialize variables to store lines and their positions
            lines = {}
            line_positions = {}
            data = pytesseract.image_to_data(image_exp, output_type=pytesseract.Output.DICT)
            # Loop through each text element
            n_boxes = len(data['text'])
            line_num = 0
            for i in range(n_boxes):
                if int(data['conf'][i]) > self.confidence_threshold and data['text'][i]:  # Confidence level filter
                    text = data['text'][i]
                    if not text.strip() or len(text.strip()) < 2 or (len(text.strip()) < 3 and data['conf'][i] < 80): continue
                    if not self.quiet:
                        print(f'Detected Text in pixel data: {data['text'][i]} at line: {line_num + 1} with confidence {data['conf'][i]}.')
                    # Group words into lines
                    if line_num not in lines:
                        lines[line_num] = [text]
                        line_positions[line_num] = {
                            'Left': data['left'][i],
                            'Top': data['top'][i],
                            'Right': data['left'][i] + data['width'][i],
                            'Bottom': data['top'][i] + data['height'][i]
                        }
                    else:
                        # if abs(line_positions[line_num]['Top'] - data['top'][i]) < 5 and abs(line_positions[line_num]['Bottom'] - (data['top'][i] + data['height'][i])) < 5:  # Same line
                        if abs(line_positions[line_num]['Top'] - data['top'][i]) < 5:
                            lines[line_num].append(text)
                            line_positions[line_num]['Left'] = min(line_positions[line_num]['Left'], data['left'][i])
                            line_positions[line_num]['Top'] = min(line_positions[line_num]['Top'], data['top'][i])
                            line_positions[line_num]['Right'] = max(line_positions[line_num]['Right'], data['left'][i] + data['width'][i])
                            line_positions[line_num]['Bottom'] = max(line_positions[line_num]['Bottom'], data['top'][i] + data['height'][i])
                        else:  # New line
                            line_num += 1
                            lines[line_num] = [text]
                            line_positions[line_num] = {
                                'Left': data['left'][i],
                                'Top': data['top'][i],
                                'Right': data['left'][i] + data['width'][i],
                                'Bottom': data['top'][i] + data['height'][i]
                            }

            for line_num, words in lines.items():
                line_text = ' '.join(words)
                position = line_positions[line_num]
                x_orig = int(position['Left'] * x_scale)
                y_orig = int(position['Top'] * y_scale)
                w_orig = int(position['Right'] * x_scale)
                h_orig = int(position['Bottom'] * y_scale)
                if not self.quiet:
                    print(f"Line {line_num}: '{line_text}' at ({x_orig}, {y_orig}, {w_orig - x_orig}, {h_orig - y_orig })")    
                box = {'Left': x_orig, 'Top': y_orig, 'Width': w_orig - x_orig, 'Height': h_orig - y_orig}
                if line_text and line_text.strip() and (len(line_text.strip()) > 2 or line_text.strip().isdigit()):
                    text_boxes.append((box, line_text))

            extracted_text = (len(text_boxes)> 0)
            if extracted_text: 
                all_ids = get_pii_boxes(text_boxes, self.pii_patterns)
        return all_ids, extracted_text

    
    def analyze_text_for_pii(self, text):
        response = self.comprehend.detect_pii_entities(Text=str(text), LanguageCode='en')
        return response['Entities']

    def analyze_text_for_phi(self, text):
        response = self.comprehend_medical.detect_phi(Text=str(text))
        return response['Entities']

    
    def save_de_id_dicom(self, local_de_id_dicom_path):
        try:
            self.ds.save_as(local_de_id_dicom_path)
            if not self.quiet:
                print(f"DICOM file has been saved to {local_de_id_dicom_path}.")
        except Exception as e:
            print(f"Error saving DICOM file {local_de_id_dicom_path}: {e}")


    def redact_tag_value(self, value, tag, vr = None):
        if value in [None, 'None', "", "none"]: return value
        """Function to replace sensitive data with placeholders or anonymous values."""
        if tag == (0x0010, 0x0020):  # Patient ID
            if value in self.patient_id_map:
                self.patient_id = self.patient_id_map[value]
                return self.patient_id
            else:
                self.patient_id = generate_patient_id(self.sequence.get("sequence"))
                self.patient_id_map[value] = self.patient_id
                return self.patient_id
        elif tag in [(0x0010, 0x0010), (0x0040, 0xa123)]:  # Patient's Name  
            return 'The^Patient'
        elif tag in [ (0x0008, 0x1060),  (0x0008, 0x0090), (0x0008, 0x1050)]:  # physician's name
            return 'Dr.^Physician'
        elif tag == (0x0008, 0x1070):  # Operator's Name
            return 'Mr.^Operator'
        elif tag == (0x0040, 0xa075):  # observer's Name
            return 'Mr.^Observer'
        elif tag == (0x0070, 0x0084): # content creator's name
            return 'Content^Creator'
        elif tag == (0x0008, 0x103e):  # Series Description
            return 'Series^Description'
        elif tag == (0x0020, 0x000d): #Study Instance UID 
            if self.ds.StudyInstanceUID in self.dicom_uid_map:
                self.studyInstanceUID = self.dicom_uid_map[self.ds.StudyInstanceUID]
                return self.dicom_uid_map[self.ds.StudyInstanceUID]
            else:
                self.studyInstanceUID = pydicom.uid.generate_uid()
                self.dicom_uid_map[self.ds.StudyInstanceUID] = self.studyInstanceUID
                return self.studyInstanceUID
        elif tag == (0x0020,0x000E): #Series Instance UID
            if self.ds.SeriesInstanceUID in self.dicom_uid_map:
                self.seriesInstanceUID = self.dicom_uid_map[self.ds.SeriesInstanceUID]
                return self.dicom_uid_map[self.ds.SeriesInstanceUID]
            else:
                self.seriesInstanceUID = pydicom.uid.generate_uid()
                self.dicom_uid_map[self.ds.SeriesInstanceUID] = self.seriesInstanceUID
                return self.seriesInstanceUID 
        elif tag == (0x0019, 0x1015): #[SlicePosition_PCS] 
            return None
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    item = EMPTY_STRING
                else:
                    if isinstance(item, float):
                        item = 0.0
                    elif isinstance(item, int):
                        item = 0
                    else:
                        item = None
            return value
        elif vr:
            if vr in ["LO", "LT", "SH", "PN", "ST", "UT", "AE"]:
                return None
            elif vr == "UI":
                if value in self.dicom_uid_map:
                    return self.dicom_uid_map[value]
                else:
                    mapped_val = pydicom.uid.generate_uid()
                    self.dicom_uid_map[value] = mapped_val
                return mapped_val
            elif vr in ["SH", "AS", "CS"]:
                if tag in [(0x0018, 0x1078), (0x0018, 0x1079)] or "DateTime" in value:
                    return  "00010101010101.000000+0000"
                return EMPTY_STRING
            elif vr in ["UL", "FL", "FD", "SL", "SS", "US"]:
                return 0
            elif vr in ["DS", "IS"]:
                return "0"
            elif vr == "UN":
                return b"ANONYMIZED"
            elif vr == "DA":
                return "00010101"
            elif vr == "DT":
                return  "00010101010101.000000+0000"
            elif vr == "TM":
                return "000000.00"
        else:
            return None
        
    def redact_id_in_image(self, all_ids):
        """
        redact id in image
        """
        # Convert the pixel data to a numpy array
        if self.pixel_array is not None:
            pixel_array = self.pixel_array
            min_pixel = np.min(pixel_array)
            max_pixel = np.max(pixel_array)
            image_8bit = Image.fromarray(self.image_data)
            draw = ImageDraw.Draw(image_8bit)
            for box in [item["Text Block"]  for item in all_ids]:
                x, y, w, h = int(box['Left']), int(box['Top']), int(box['Width']), int(box['Height'])
                draw.rectangle([x, y, x + w, y + h], fill="white")

            if self.pixel_array.dtype != np.uint8:
                # Re-normalize to the original 16-bit range
                if hasattr(self.ds, 'PixelRepresentation') and self.ds.PixelRepresentation == 1 and hasattr(self.ds, 'WindowCenter') and hasattr(self.ds, 'WindowWidth'):
                    drawn_array_16bit = reverse_windowing(np.array(image_8bit), self.ds)
                elif min_pixel < 0: #signed int16
                    image_float = np.array(np.array(image_8bit)).astype(np.float32) / 255
                    # Scale to int16 range
                    drawn_array_16bit = image_float * (max_pixel - min_pixel * -1) + min_pixel
                    # Clip the values to the int16 range to prevent overflow
                    drawn_array_16bit = np.clip(drawn_array_16bit, -32768, 32767).astype(np.int16)
                else:
                    drawn_array_16bit = (np.array(np.array(image_8bit)).astype(np.float32) / 255 * (max_pixel - min_pixel) + min_pixel).astype(np.uint16)
                # Combine the drawn areas with the original image to avoid overwriting non-drawn areas
                pixel_array = np.maximum(pixel_array, drawn_array_16bit)
            else:
                pixel_array = np.array(image_8bit)

        # Update the DICOM dataset with the new pixel data
        if pixel_array is not None:
            self.ds.PixelData = pixel_array.tobytes()
        # return generate_clean_image(image, [ item["Text Block"]  for item in all_ids], output_image_path)
        return True
    
    def update_rules_in_configs(self, config_file = None):
        """
        Update rules and export to YAML file.
        """
        # Process the dictionary
        tags_key = "dicom_tags"
        if not config_file: 
            config_file = self.rule_config_file_path
        dictionary = {"rules": {
            tags_key: process_dict_tags(self.dicom_tags, "tag"), 
            "keywords": self.sensitive_words, "regex": list(self.regex), 
            "confidence_threshold": self.confidence_threshold,  
        }}
        if not self.quiet:
            print("Updated rules:")
            print(dictionary)
        # save change to the rules config file.
        dict2yaml(dictionary, config_file)

    def save_mappings(self):
        """
        Save mappings to JSON files.
        """
        if self.dicom_uid_map:
            save_dict_to_json(self.dicom_uid_map, DICOM_UID_MAP_JSON)
            convert_json_to_csv(DICOM_UID_MAP_JSON, DICOM_UID_MAP_JSON.replace(".json", ".csv"))
        if self.sequence:
            save_dict_to_json(self.sequence, PATIENT_SEQUENCES_JSON)
        if self.patient_id_map:
            save_dict_to_json(self.patient_id_map, PATIENT_ID_MAP_JSON)
            convert_json_to_csv(PATIENT_ID_MAP_JSON, PATIENT_ID_MAP_JSON.replace(".json", ".csv"))

        # convert json to csv
    

    def close(self):
        self.save_mappings()
        """
        Close the DICOM dataset and release resources.
        """
        if self.ds:
            self.ds = None
        if self.boto3_session:
            self.boto3_session = None
        if self.rekognition:
            self.rekognition = None
        if self.comprehend_medical:
            self.comprehend_medical = None
        if self.s3_client:
            self.s3_client = None

        try:
            sys.exit(1)
        except: 
            print("Completed!")
   
def generate_patient_id(sequence_list):
    """
    Generate a unique patient ID based on a sequence number.
    """
    sequence = 0
    if len(sequence_list) == 0:
        sequence = 1
        sequence_list.append(sequence)
    else:
        sequence = sequence_list[-1] + 1
        sequence_list.append(sequence)
    # Generate a unique patient ID
    patient_id = f"patient^{sequence:04d}"
    return patient_id

def inspect_dicom_file(dicom_dataset):
    """
    Inspect DICOM file and extract sensitive information.
    """

    # Check if 'Bits Stored' (0028,0101) is present in the dataset
    if 'BitsStored' in dicom_dataset:
        bits_stored = dicom_dataset.BitsStored
        print(f"Bits Stored: {bits_stored}")
    else:
        print("'Bits Stored' (0028,0101) tag is not present in this DICOM file.")
    print(dicom_dataset)
    # Inspect pixel data type and shape
    pixel_data = dicom_dataset.pixel_array
    print(f"Pixel Data Type: {pixel_data.dtype}")
    print(f"Pixel Data Shape: {pixel_data.shape}")



        