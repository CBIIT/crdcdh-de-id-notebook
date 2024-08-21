import pydicom, io
import pytesseract
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from botocore.exceptions import ClientError
from common.utils import process_dict_tags, get_date_time, dict2yaml, get_boto3_session, yaml2dict, generate_regex, convert_basetag
from common.sagemaker_utils import get_sagemaker_execute_role, create_local_output_dirs
from common.de_id_utils import generate_clean_image, get_pii_boxes

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
        self.dicom_tags = None
        self.phi_tags = None
        self.sensitive_words = None
        self.vr = None
        self.regex = None
        self.confidence_threshold = None
        self.set_rules(rule_config_file_path)
        # dicom data
        self.ds = None
        self.pixel_array = None
        self.image_data = None

    def set_rules(self, rule_config_file_path):
        """
        set rules for DICOM tags and keywords
        """
        # Load the rules from the YAML file
        self.rules = yaml2dict(rule_config_file_path)["rules"]
        self.dicom_tags = self.rules['dicom_tags']
        self.phi_tags = [ tuple(item["tag"]) for item in self.dicom_tags ]
        # print(self.phi_tags)
        self.vr = self.rules['vr']
        self.sensitive_words = self.rules['keywords']
        # print(self.sensitive_words)
        self.regex = self.rules['regex']
        self.confidence_threshold = int(self.rules['confidence_threshold'])
        self.pii_patterns = re.compile(r'\b(?:{0})\b'.format('|'.join(self.regex)))


    def parse_dicom_file(self, bucket, key, local_dicom_path, useAI = False):
        """
        parse dicom file and upload DICOM file to s3 bucketfor de-identification.
        :param src_bucket: source s3 bucket
        :param src_key: source s3 key
        :param local_dicom_path: local dicom file path
        :return: True if the upload is successful, False otherwise
        :rtype: bool, pydicom.dataset.FileDataset
        """

        try:
            # Load the DICOM data
            self.ds = pydicom.dcmread(local_dicom_path)
            # Extract pixel array and convert to uint8 (if necessary)
            self.pixel_array = self.ds.pixel_array
            np.seterr(divide='ignore', invalid='ignore')
            image_data = None
            if not useAI:
                if self.ds.BitsAllocated == 16:
                    image_data = self.pixel_array.astype(np.uint16)
                else:
                    image_data = self.pixel_array.astype(np.uint8)
            else:
                if self.pixel_array.dtype != np.uint8:
                    image_data = (self.pixel_array / np.max(self.pixel_array) * 255).astype(np.uint8)
                else:
                    image_data = self.pixel_array.astype(np.uint8)
                # Convert the pixel array to a PIL Image
                image = Image.fromarray(image_data)
                image_io = io.BytesIO()
                image.save(image_io, format='PNG')
                image_io.seek(0)
                self.s3_client.upload_fileobj(image_io, bucket, key)
                # initial AI foe evaluation
                self.rekognition = self.boto3_session.client('rekognition')
                self.comprehend_medical = self.boto3_session.client('comprehendmedical')

            self.image_data = image_data
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
        # Redact PHI in the DICOM dataset
        redacted = 0
        redacted_value = None
        detected_tags = []
        for item in self.ds:
            value = item.value
            if item.value in [None, 'None', "", "none"]:  continue
            vr = item.VR
            name = item.name
            tuple = (item.tag.group, item.tag.element)
            if vr in self.vr:
                redacted_value = self.redact_tag_value(value, tuple)
                if not self.quiet:
                    print(f"Tag: {item} - Redacted Value: {redacted_value}")
                item.value = redacted_value
                detected_tags.append(tuple)
                redacted += 1
                continue
            in_keywords = [key for key in self.sensitive_words if key in name]
            if len(in_keywords) > 0:
                redacted_value = self.redact_tag_value(value, tuple)
                if not self.quiet:
                    print(f"Tag: {item} - Redacted Value: {redacted_value}")
                item.value = redacted_value
                detected_tags.append(tuple)
                redacted += 1
                continue
            if tuple in self.phi_tags:
                redacted_value = self.redact_tag_value(value, tuple)
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
        for item in self.ds:
            # Detect PHI/PII in the text
            if not item.value: pass
            if isinstance(item.value, str):
                ids = self.detect_id_in_text_AI(item.value, is_image=False)
                if ids and len(ids) > 0:
                    if not self.quiet:
                        print(f"Tag: {item}")
                    all_ids.extend(ids)
                    detected_elements.append(item)
                    tuple = (item.tag.group, item.tag.element)
                    tags.append(tuple)
                    self.dicom_tags.append({"tag": tuple, "name": item.name})
            elif isinstance(item.value, list):
                for item in item.value:
                    if not item: pass
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if not value: pass
                            ids = self.detect_id_in_text_AI(value, is_image=False)
                            ids_list.extend(ids)
                            if ids and len(ids) > 0 and not self.quiet:
                                print(f"Tag: {item}")
                    else:
                        ids = self.detect_id_in_text_AI(item, is_image=False)
                        ids_list.extend(ids)
                        if ids and len(ids) > 0:
                            print(f"Tag: {item}")
                if len(ids_list) > 0:
                    if not self.quiet:
                        print(f"Tag: {item}")
                    detected_elements.append(item)
                    tuple = (item.tag.group, item.tag.element)
                    tags.append(tuple)
                    self.dicom_tags.append({"tag": tuple, "name": item.name})
                    
        return detected_elements, tags, all_ids, 
    
    def redact_tags(self, elements):
        """
        redact PHI info in DICOM by Comprehend Medical
        """
        for item in elements:
            value = item.value
            if value in [None, 'None', "", "none"]:  continue
            item.value = self.redact_tag_value(value, (item.tag.group, item.tag.element))
    
    def detect_id_in_text_AI(self, detected_text, is_image = False):
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
                if entity['Score'] >= self.confidence_threshold/100: 
                    valid_entities.append(entity)
                    if not self.quiet:
                        print(f"Entity: {entity['Text']} - Type: {entity['Type']} - Confidence: {entity['Score']}")
                    if is_image and entity['Type'] in ["ID", "AGE", "ADDRESS", "PHONE_OR_FAX", "DATE"]:
                        regex = generate_regex(entity['Text'])
                        if regex:
                            self.regex.append(regex)
            if len(valid_entities) > 0:
                ids.append(text)
        return ids
    
   
    
    def detect_id_in_img(self, bucket_name, image_key,  use_AI= False):
        """
        detect phi in image with or without AI
        """
        all_ids = []
        extracted_text = False
        if use_AI:
            # Use Rekognition to detect text
            if not self.rekognition:
                self.rekognition = self.boto3_session.client('rekognition')
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
            image = Image.fromarray(self.image_data)
            text_boxes = []
            # Initialize variables to store lines and their positions
            lines = {}
            line_positions = {}
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            # Loop through each text element
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > self.confidence_threshold and data['text'][i]:  # Confidence level filter
                    text = data['text'][i]
                    if not text or not text.strip() or not (len(text.strip()) > 2 or text.strip().isdigit()): continue
                    if not self.quiet:
                        print(f'Detected Text in pixel data: {data['text'][i]} with confidence {data['conf'][i]}.')
                    line_num = data['line_num'][i]
                    word = data['text'][i]
                    # Group words into lines
                    if line_num not in lines:
                        lines[line_num] = []
                        line_positions[line_num] = {
                            'Left': data['left'][i],
                            'Top': data['top'][i],
                            'Right': data['left'][i] + data['width'][i],
                            'Bottom': data['top'][i] + data['height'][i]
                        }
                    else:
                        line_positions[line_num]['Right'] = max(line_positions[line_num]['Right'], data['left'][i] + data['width'][i])
                        line_positions[line_num]['Bottom'] = max(line_positions[line_num]['Bottom'], data['top'][i] + data['height'][i])
                
                    lines[line_num].append(word)

            for line_num, words in lines.items():
                line_text = ' '.join(words)
                position = line_positions[line_num]
                # print(f"Line {line_num}: '{line_text}' at ({position['Left']}, {position['Top']}, {position['Right'] - position['Left']}, {position['Bottom'] - position['Top']})")    
                box = {'Left': position['Left'], 'Top': position['Top'], 'Width': position['Right'] - position['Left'], 'Height': position['Bottom'] - position['Top']}
                if line_text and line_text.strip() and (len(line_text.strip()) > 2 or line_text.strip().isdigit()):
                    text_boxes.append((box, line_text))

            extracted_text = (len(text_boxes)> 0)
            if extracted_text: 
                # use regular expression and name parser to detect PHI in text
                all_ids = get_pii_boxes(text_boxes, self.pii_patterns)

        return all_ids, extracted_text

    
    def analyze_text_for_pii(self, text):
        response = self.comprehend.detect_pii_entities(Text=str(text), LanguageCode='en')
        return response['Entities']

    def analyze_text_for_phi(self, text):
        response = self.comprehend_medical.detect_phi(Text=str(text))
        return response['Entities']

    
    def save_de_id_dicom(self, local_de_id_dicom_path):
        # if redacted_img_path:
        #     # Load the PNG image
        #     image = Image.open(redacted_img_path)
        #     # # Convert the image to a NumPy array
        #     image_array = np.array(image)
        #     if ds.BitsAllocated == 16:
        #         min_pixel = np.min(ds.pixel_array)
        #         max_pixel = np.max(ds.pixel_array)
        #         image_array = (np.array(image).astype(np.float32) / 255 * (max_pixel - min_pixel) + min_pixel).astype(np.uint16)
        #         image_array = image_array * 256  # Scale up 8-bit drawing to 16-bit
        #         # Update the modified pixel data in the DICOM dataset
        #         image_array = np.maximum(ds.pixel_array, image_array)
        #     # Add the image data to the dataset
        #     ds.Rows, ds.Columns = image_array.shape
        #     ds.PixelData = image_array.tobytes()
        # Save the DICOM file
        self.ds.save_as(local_de_id_dicom_path)
        if not self.quiet:
            print(f"DICOM file has been saved to {local_de_id_dicom_path}.")


    def redact_tag_value(self, value, tag):
        if value in [None, 'None', "", "none"]: return value
        """Function to replace sensitive data with placeholders or anonymous values."""
        if tag in [(0x0010, 0x0010), (0x0040, 0xa123)]:  # Patient's Name  
            return 'The^Patient'
        elif tag in [ (0x0008, 0x1060),  (0x0008, 0x0090), (0x0008, 0x1050)]:  # physician's name
            return 'Dr.^Physician'
        elif tag == (0x0008, 0x1070):  # Operator's Name
            return 'Mr.^Operator'
        elif tag == (0x0040, 0xa075):  # observer's Name
            return 'Mr.^Observer'
        elif isinstance(value, list):
            return [None for _ in value]
        else:
            return None

    def redact_id_in_image(self, all_ids):
        """
        redact id in image
        """
        # Convert the pixel data to a numpy array
        pixel_array = self.pixel_array

        if self.ds.BitsAllocated == 16:
            # Normalize the 16-bit image to the full 8-bit range (0-255)
            min_pixel = np.min(pixel_array)
            max_pixel = np.max(pixel_array)
            normalized_8bit = ((pixel_array - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
        else:
            normalized_8bit = pixel_array.astype(np.uint8)

        image_8bit = Image.fromarray(normalized_8bit)
        draw = ImageDraw.Draw(image_8bit)
        for box in [item["Text Block"]  for item in all_ids]:
            x, y, w, h = int(box['Left']), int(box['Top']), int(box['Width']), int(box['Height'])
            draw.rectangle([x, y, x + w, y + h], fill="white")

        if self.ds.BitsAllocated == 16:
            # Re-normalize to the original 16-bit range
            drawn_array_16bit = (np.array(image_8bit).astype(np.float32) / 255 * (max_pixel - min_pixel) + min_pixel).astype(np.uint16)
            # Combine the drawn areas with the original image to avoid overwriting non-drawn areas
            pixel_array = np.maximum(pixel_array, drawn_array_16bit)
        else:
            pixel_array = np.array(image_8bit)

        # Update the DICOM dataset with the new pixel data
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
        dictionary = {}
        dictionary["rules"] = self.rules
        dictionary["rules"][tags_key] = self.dicom_tags
        dictionary["rules"]["vr"] = self.vr
        dictionary["rules"]["keywords"] = self.sensitive_words
        dictionary["rules"]["regex"] = self.regex
        dictionary["rules"]["confidence_threshold"] = self.confidence_threshold
        dictionary["rules"][tags_key] = process_dict_tags( dictionary["rules"][tags_key], "tag")
        print(dictionary)

        # save change to the rules config file.
        dict2yaml(dictionary, config_file)


        