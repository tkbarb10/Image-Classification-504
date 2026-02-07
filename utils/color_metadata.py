import numpy as np

def color_metadata(array, colors=['r', 'g', 'b']):    
    metadata = {}
    for i, color in enumerate(colors):
        channel = array[:, :, i]
        metadata[f"{color}_sum"] = channel.sum()
        metadata[f"{color}_mean"] = channel.mean()
        metadata[f"{color}_std"] = channel.std()
        metadata[f"{color}_median"] = np.median(channel)
        metadata[f"{color}_max"] = channel.max()
        metadata[f"{color}_min"] = channel.min()
    return metadata