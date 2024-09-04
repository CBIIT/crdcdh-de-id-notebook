from PIL import ImageDraw
import re
from nameparser import HumanName
from common.constants import HUMAN_NAME_REGEX, REPEAT_CHAR_REGEX, KEYWORDS_IN_PIXEL_JSON
from common.utils import load_json_file

keywords = load_json_file(KEYWORDS_IN_PIXEL_JSON).get('keywords', [])

def contains_name(text):
    # Remove any leading/trailing whitespace
    text = text.strip()
    if bool(re.search(REPEAT_CHAR_REGEX, text)) or any(keyword in text for keyword in keywords) or text.startswith('-') or text.startswith('_'):
        # print("Found invalid name: " + text)
        return False
    if len(text.split(" ")) < 2: return False
    # remove any non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Parse the text as a potential human name
    name = HumanName(text)
    # Check if the parsed result contains a valid name
    if name.first and re.match( r'^[A-Z]', name.first) and name.last and re.match( r'^[A-Z]', name.first) and not (len(name.last) < 3 and re.match(r'([a-zA-Z])\1', name.last)):
        # print("Found name: " + name.first + " " + name.last)
        matches = re.findall(HUMAN_NAME_REGEX, name.first + " " + name.last)
        return len(matches) > 0 
    else: return False

def generate_clean_image(image, boxes, output_image_path):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x, y, w, h = int(box['Left']), int(box['Top']), int(box['Width']), int(box['Height'])
        draw.rectangle([x, y, x + w, y + h], fill="red")
    image.save(output_image_path)
    return True

def load_image(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()
    
def get_text_boxes(detected_txt):
    text_detections = detected_txt['TextDetections']
    boxes = []
    for text in text_detections:
        if text['Type'] == 'LINE':
            box = text['Geometry']['BoundingBox']
            boxes.append(box)
    return boxes

def is_pii(text, pii_patterns):
    # 1) check if the text contain human names
    result = contains_name(text)
    # Find all matches
    matches = pii_patterns.findall(text)
    # for match in matches:
    # print(f"Detected PHI: {matches}")
    if result or len(matches) > 0:
        return True
    return False

def get_pii_boxes(text_blocks, pii_patterns):
    pii_boxes = []
    for box, text in text_blocks:
        if not text or text.strip() == "":
            continue
        # print("Found text line: " + text)
        result = is_pii(text, pii_patterns)
        if result:
            pii_boxes.append({"Text": text, "Text Block": box})
    return pii_boxes

