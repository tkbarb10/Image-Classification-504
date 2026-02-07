from PIL import Image
import os
import numpy as np

def read_image_pillow(path):
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            arr = np.array(img)
            return arr
    except Exception as e:
        return None 