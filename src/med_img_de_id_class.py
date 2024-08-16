import pydicom
import pytesseract
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from botocore.exceptions import ClientError
from common.utils import process_dict_tags, get_date_time, dict2yaml, get_boto3_session, yaml2dict
from common.sagemaker_utils import get_sagemaker_execute_role, create_local_output_dirs
from common.de_id_utils import generate_clean_image, get_pii_boxes

class ProcessMedImage:
    def __init__(self, rule_config_file_path= '../configs/de-id/de_id_rules_auto.yaml'):
        """
        constructor of ProcessMedImage
        """
        create_local_output_dirs(["../output", "../output/model", "../data", "../data/raw", "../data/raw/json","../temp"]) # create local directories if not exist.
        self.endpoint_name = None
        self.boto3_session = get_boto3_session()
        self.timestamp = get_date_time()
        self.data_time = get_date_time("%Y-%m-%d-%H-%M-%S")
        self.role = get_sagemaker_execute_role(self.boto3_session)
        self.region_name = self.boto3_session.region_name
        self.s3_client = self.boto3_session.client('s3')
        self.rekognition= None
        self.comprehend = None
        self.comprehend_medical = None
        #Define the S3 bucket and object for the medical image we want to analyze.
        self.phi_detection_threshold = 0.00
        self.local_img_folder = "../images/med_phi_img/"
        self.rule_config_file_path = rule_config_file_path
        self.dicom_tags = None
        self.phi_tags = None
        self.sensitive_words = None
        self.regex = None
        self.set_rules(rule_config_file_path)
        self.image_data = None

    def set_rules(self, rule_config_file_path):
        """
        set rules for DICOM tags and keywords
        """
        # Load the rules from the YAML file
        self.rules = yaml2dict(rule_config_file_path)["rules"]
        self.dicom_tags = self.rules['dicom_tags']
        self.phi_tags = [ tuple(item["tag"]) for item in self.rules['dicom_tags']]
        # print(self.phi_tags)
        self.sensitive_words = self.rules['keywords']
        # print(self.sensitive_words)
        self.regex = self.rules['regex']
        self.pii_patterns = re.compile(r'\b(?:{0})\b'.format('|'.join(self.regex)))


    def upload_dicom_file(self, src_bucket, src_key, local_dicom_path, useAI = False):
        """
        parse dicom file and upload DICOM file to s3 bucketfor de-identification.
        :param src_bucket: source s3 bucket
        :param src_key: source s3 key
        :param local_dicom_path: local dicom file path
        :return: True if the upload is successful, False otherwise
        :rtype: bool, pydicom.dataset.FileDataset
        """

        try:
            # upload DICOM file to source s3 bucket
            self.s3_client.upload_file(local_dicom_path, src_bucket, src_key)
            print("DICOM file has been uploaded to the source a3 bucket.")

            # Load the DICOM data
            dicom_dataset = pydicom.dcmread(local_dicom_path)

            # Extract pixel array and convert to uint8 (if necessary)
            image_data = dicom_dataset.pixel_array
            self.image_data = image_data

            if image_data.dtype != np.uint8:
                self.image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)

            # Convert the pixel array to a PIL Image
            image = Image.fromarray(image_data)

            # Save the pixel data to png file
            png_image_path = local_dicom_path.replace('.dcm', '.png')
            image.save(png_image_path) 
            
            if useAI:
                self.s3_client.upload_file(png_image_path, src_bucket, src_key.replace('.dcm', '.png'))

            return True, dicom_dataset
        except ClientError as ce:
            print(f"Error uploading DICOM file: {ce}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def de_identify_dicom(self, ds):
        """
        de-identify DICOM metadata with HIPPA Privacy Rules
        """
        print("De-identifying DICOM metadata")
        # Redact PHI in the DICOM dataset
        redacted_value = None
        #  redact date and UID tags
        detected_tags = []
        for item in ds:
            name = item.name
            for key in self.sensitive_words:
                if key in name:
                    redacted_value = self.redact_tag_value(name, item.value, item.tag)
                    # print(f"Tag: {item.tag} - {name} - Value: {item.value} - Redacted Value: {redacted_value}")
                    print(f"Tag: {item} - Redacted Value: {redacted_value}")
                    item.value = redacted_value
                    detected_tags.append(item.tag)
        for tag in self.phi_tags:
            if tag in ds and not tag in detected_tags:
                name = ds[tag].name
                ds_tag = ds[tag]
                redacted_value = self.redact_tag_value(name, ds[tag].value, ds_tag)
                print(f"Tag: {ds_tag} - Redacted Value: {redacted_value}")
                ds[tag].value = redacted_value

        print(f"Redacted DICOM matadata")

    def detect_id_in_tags(self, dicom_data):
        """
        detect PHI info in DICOM by Comprehend Medical
        """
        tags = []
        all_ids = []
        ids_list = []
        ids = []
        self.comprehend= self.boto3_session.client('comprehend')
        self.comprehend_medical = self.boto3_session.client('comprehendmedical')
        for elem in dicom_data:
            tags.append(elem.tag)
            # Detect PHI/PII in the text
            # print(f"Tag: {elem.tag} - Name: {elem.name} - Value: {elem.value}")
            if not elem.value: pass
            if isinstance(elem.value, str):
                ids = self.detect_id_in_text_AI(elem.value, is_image=False)
                if ids and len(ids) > 0:
                    print(f"Tag: {elem}")
                    all_ids.extend(ids)
                    tuple = (elem.tag.group, elem.tag.element)
                    print(f"tag: {tuple}, Name:  {elem.name}")
                    self.dicom_tags.append({"tag": tuple, "name": elem.name})
            elif isinstance(elem.value, list):
                for item in elem.value:
                    if not item: pass
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if not value: pass
                            ids = self.detect_id_in_text_AI(value, is_image=False)
                            ids_list.extend(ids)
                            if ids and len(ids) > 0:
                                print(f"Tag: {elem}")
                    else:
                        ids = self.detect_id_in_text_AI(item, is_image=False)
                        ids_list.extend(ids)
                        if ids and len(ids) > 0:
                            print(f"Tag: {elem}")
                if len(ids_list) > 0:
                    print(f"Tag: {elem}")
                    tuple = (elem.tag.group, elem.tag.element)
                    print(f"tag: {tuple}, Name:  {elem.name}")
                    self.dicom_tags.append({"tag": tuple, "name": elem.name})
                    
        return tags, all_ids
    
    def redact_tags(self, dicom_data, tags):
        """
        redact PHI info in DICOM by Comprehend Medical
        """
        for tag in tags:
            if tag in dicom_data:
                dicom_data[tag].value = self.redact_tag_value(dicom_data[tag].name, dicom_data[tag].value, dicom_data[tag])
    
    def detect_id_in_text_AI(self, detected_text, is_image = False):
        # Step 4: Analyze text blocks for PHI using Comprehend Medical
        ids = []
        if not detected_text: return ids
        text = detected_text['DetectedText'] if is_image else detected_text
        phi_entities = self.analyze_text_for_phi(text)
        if phi_entities and len(phi_entities) > 0:
            print(f"PHI detected in text:")
            for entity in phi_entities:
                print(f"Entity: {entity['Text']} - Type: {entity['Type']} - Confidence: {entity['Score']}")
            ids.append(text)
        return ids
    
    def draw_img(self, img, dpi = 72):
        #Set the image color map to grayscale, turn off axis graphing, and display the image
        plt.rcParams["figure.figsize"] = [16,9]
        # Display the image
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('DICOM Image')
        plt.axis('off')  # Turn off the axis
        plt.show()
    
    def detect_id_in_img(self, local_png_path, bucket_name, image_key,  use_AI= False):
        """
        detect phi in image with or without AI
        """
        all_ids = []
        if use_AI:
            # Use Rekognition to detect text
            self.rekognition = self.boto3_session.client('rekognition')
            # Load the saved image and convert to bytes
            image = Image.open(local_png_path)

            # Detect text in the image using Rekognition
            response = self.rekognition.detect_text(Image={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': image_key
                }
            })
            detected_texts = [text for text in response['TextDetections']]
            img_width, img_height = image.size
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
                if int(data['conf'][i]) > 0 and data['text'][i]:  # Confidence level filter
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
                text_boxes.append((box, line_text))

            # use regular expression and name parser to detect PHI in text
            all_ids = get_pii_boxes(text_boxes, self.pii_patterns)

        return all_ids

    
    def analyze_text_for_pii(self, text):
        response = self.comprehend.detect_pii_entities(Text=str(text), LanguageCode='en')
        return response['Entities']

    def analyze_text_for_phi(self, text):
        response = self.comprehend_medical.detect_phi(Text=str(text))
        return response['Entities']

    
    def save_de_id_dicom(self, redacted_img_path, ds, local_de_id_dicom_path):
       # Load the PNG image
        image = Image.open(redacted_img_path)
        image = image.convert('L')
        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Add the image data to the dataset
        ds.Rows, ds.Columns = image_array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelData = image_array.tobytes()

        # Save the DICOM file
        ds.save_as(local_de_id_dicom_path)

        print(f"DICOM file has been created and saved to {local_de_id_dicom_path}.")


    def redact_tag_value(self, name, value, tag):
        """Function to replace sensitive data with placeholders or anonymous values."""
        if tag == (0x0010, 0x0010):  # Patient's Name
            return 'The^Patient'
        elif tag == (0x0008, 0x1060) or tag == (0x0008, 0x0090):  # physician's name
            return 'Dr.^Physician'
        elif tag == (0x0008, 0x1070):  # Operator's Name
            return 'Mr.^Operator'
        # elif isinstance(value, bool):
        #     return None
        elif isinstance(value, str):
            return None
        # elif isinstance(value, (int, float)):
        #     return None
        # elif isinstance(value, (datetime)):
        #     return None
        elif isinstance(value, list):
            return [None for _ in value]
        else:
            return None
        return value

    def redact_id_in_image(self, all_ids, output_image_path):
        """
        redact id in image
        """
        image = Image.fromarray(self.image_data)

        return generate_clean_image(image, [ item["Text Block"]  for item in all_ids], output_image_path)

    
    def update_rules_in_configs(self, tags, keywords, config_file = None):
        """
        Convert dictionary to YAML and save to file.
        """
        # Process the dictionary
        tags_key = "dicom_tags"
        if not config_file: 
            config_file = self.rule_config_file_path
        dictionary = {}
        dictionary["rules"] = self.rules
        dictionary["rules"][tags_key] = tags
        dictionary["rules"]["keywords"] = keywords
        dictionary["rules"][tags_key] = process_dict_tags( dictionary["rules"][tags_key], "tag")
        print(dictionary)

        # save change to the rules config file.
        dict2yaml(dictionary, config_file)

        