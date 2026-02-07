import numpy as np
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
from skimage.filters import sobel

# Sobel is measuring change in intensity and is designed for single channel images (since color change isn't what's being measured)

def edges(array):
    edges_metadata = {}
    gray_array=rgb2gray(array)
    edges=sobel(gray_array)
    edges_metadata['gray_mean']=gray_array.mean() # average brightness scaled 0-1
    edges_metadata['gray_std']=gray_array.std()
    edges_metadata['edge_mean']=edges.mean() # average edge strength scaled 0-1
    edges_metadata['edge_std']=edges.std()
    edges_metadata['edge_median']=np.median(edges)
    edges_metadata['edge_entropy']=shannon_entropy(edges)
    edges_metadata['edge_density']=(edges > .05).mean()

    return edges_metadata