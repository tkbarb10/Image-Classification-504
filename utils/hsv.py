import numpy as np
from scipy.stats import mode
from skimage.color import rgb2gray, rgb2hsv

def hsv(array):
    hsv_metadata = {}
    hsv_array = rgb2hsv(array)
    hue = hsv_array[..., 0] # the distribution of colors
    sat = hsv_array[..., 1] # how rich the colors are
    mask = sat > .2
    hsv_metadata['hue_mode']=mode(hue).mode[0] # what color dominates
    hsv_metadata['hue_mode_threshold']=mode(hue[mask], axis = None).mode if np.any(mask) else np.nan # what color dominates from colors with saturation (distinguish from background)
    hsv_metadata['sat_mean']=sat.mean() # overall color intensity
    hsv_metadata['sat_std']=sat.std() # variation of "dull" and 'vivid'

    return hsv_metadata
    