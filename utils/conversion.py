import pandas as pd
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from scipy.stats import mode
from .read_image import read_image_pillow
from .color_metadata import color_metadata
from .gray_edges_metadata import edges
from .hsv import hsv

def convert(paths, df, max_workers=16):

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        arrays = list(
            tqdm(
                executor.map(
                    lambda p: read_image_pillow(p),
                    paths
                ),
                total=len(paths)
            )
        )

    print("Number of loaded arrays:", sum([arr is not None for arr in arrays]))
    print("Number of failed arrays:", sum([arr is None for arr in arrays]))

    failed_arrays = [i for i, arr in enumerate(arrays) if arr is None]
            
    arrays = [arr for arr in arrays if arr is not None] 
    
    if failed_arrays:
        df = df.drop(index=failed_arrays).reset_index(drop=True)

    heights = [arr.shape[0] for arr in arrays]
    widths = [arr.shape[1] for arr in arrays]

    df['height'] = heights
    df['width'] = widths

    with ProcessPoolExecutor(max_workers=4) as executor:
        color_channel_metadata = list(tqdm(executor.map(color_metadata, arrays), total=len(arrays)))
        edges_metadata = list(tqdm(executor.map(edges, arrays), total=len(arrays)))
        hsv_metadata = list(tqdm(executor.map(hsv, arrays), total=len(arrays)))

    all_metadata = [
        {**color_dict, **edges_dict, **hsv_dict}
        for color_dict, edges_dict, hsv_dict in zip(color_channel_metadata, edges_metadata, hsv_metadata)
    ]
    metadata_df=pd.DataFrame(all_metadata)

    df = pd.concat([df.reset_index(drop=True), metadata_df.reset_index(drop=True)], axis=1)
    return df












    
        