import re
import numpy as np
from .read_image import read_image_pillow

def image_histogram_features(image_path):
    
    image_name = re.sub(r".*[\\/](.*?)\.png.*", r"\1", image_path)
    image_array = read_image_pillow(image_path)
    
    r, _ = np.histogram(image_array[:, :, 0], bins=256, range=(0,256))
    g, _ = np.histogram(image_array[:, :, 1], bins=256, range=(0,256))
    b, _ = np.histogram(image_array[:, :, 2], bins=256, range=(0,256))
    
    # Create column names for each bin
    feature_dict = {'image_name': image_name}
    feature_dict.update({f'r_{i}': r[i] for i in range(256)})
    feature_dict.update({f'g_{i}': g[i] for i in range(256)})
    feature_dict.update({f'b_{i}': b[i] for i in range(256)})
    
    return feature_dict