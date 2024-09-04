import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance

# pixel utils
def parse_pixel_data(ds, max_pixel, min_pixel):
    # convert to uint8 for extract test and block PHI info in pixel data
    image_data = None
    pixel_array = None
    number_of_frames = ds.get('NumberOfFrames', 1) 
    pixel_array = ds.pixel_array[0] if number_of_frames > 1 else ds.pixel_array
    # bitsStored = ds.get('BitsStored', 16)
    # if bitsStored == 10:
    #     ds.BitsStored = 16
    if hasattr(ds, "PixelRepresentation") and ds.PixelRepresentation == 1:
        max_pixel += min_pixel * -1
        min_pixel = 0
        pixel_data_shifted = pixel_array + min_pixel * -1
        # print(f"Max: {max_pixel}, dtype: {pixel_data_shifted.dtype}")
        # Normalize to 0-255 range
        if pixel_array.dtype != np.uint8:
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                image_data = apply_windowing(pixel_data_shifted, ds)
            else: 
                image_data = ((pixel_data_shifted /max_pixel) * 255).astype(np.uint8)
        else:
            image_data = pixel_data_shifted.astype(np.uint8)
    else: 
        if ds.pixel_array.dtype != np.uint8:
            if min_pixel < 0: #signed int16
                # Shift the int16 values to be in the range of 0-65535
                image_shifted = image_data.astype(np.int32) + 32768
                # Scale to the range of 0-255
                image_data = (image_shifted / 65535 * 255).astype(np.uint8)
            else:
                image_data = ((pixel_array / np.max(pixel_array) * 255).astype(np.float32)).astype(np.uint8)
        else:
            image_data = pixel_array.astype(np.uint8)

    return image_data

def apply_windowing(pixel_data, ds):
    window_center = ds.WindowCenter if isinstance(ds.WindowCenter, (int, float)) else ds.WindowCenter[0]
    window_width = ds.WindowWidth if isinstance(ds.WindowWidth, (int, float)) else ds.WindowWidth[0]
    # Calculate the minimum and maximum pixel values for the window
    min_value = window_center - (window_width / 2)
    max_value = window_center + (window_width / 2)

    # Clip the pixel data to the window range
    pixel_data_clipped = np.clip(pixel_data, min_value, max_value)
    
    # Normalize the pixel data to 0-255 range
    pixel_data_normalized = (pixel_data_clipped - min_value) / (max_value - min_value) * 255

    return pixel_data_normalized.astype(np.uint8)

def reverse_windowing(windowed_pixel_data, ds):

    window_center = ds.WindowCenter if isinstance(ds.WindowCenter, (int, float)) else ds.WindowCenter[0]
    window_width = ds.WindowWidth if isinstance(ds.WindowWidth, (int, float)) else ds.WindowWidth[0]
    min_val = window_center - 0.5 - (window_width - 1) / 2
    max_val = window_center - 0.5 + (window_width - 1) / 2
    
    # Scale the uint8 image back to the int16 range
    int16_pixel_data = windowed_pixel_data.astype(np.float32) / 255 * (max_val - min_val) + min_val
    return int16_pixel_data.astype(np.int16)

def enhance_image(image_data, local_dicom_path, need_contrast=False):
    try:
        image = Image.fromarray(image_data)
        if need_contrast:
            # Enhance image contrast
            image = ImageEnhance.Contrast(image)
            image = image.enhance(2)  # Adjust contrast factor as needed
            threshold = 128  # Adjust threshold as needed
            image = image.point(lambda p: p > threshold and 255)
            image = image.convert('L')    
        image = image.filter(ImageFilter.SHARPEN)
        return image
    except Exception as e:
        print(f"Error converting pixel data to image in {"/".join(local_dicom_path.split("/")[5:])}: {e}")
        return None