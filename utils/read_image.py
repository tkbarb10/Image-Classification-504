import ast
import io

import numpy as np
from PIL import Image


def read_image_pillow(path):
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            arr = np.array(img)
            return arr
    except Exception as e:
        return None


def read_image_from_bytes(byte_string: str) -> np.ndarray | None:
    """Read an image array from a string representation of bytes.

    Handles the case where image bytes have been stored as a string
    (e.g. from a parquet column) and need to be evaluated back to
    actual bytes before decoding.

    Parameters
    ----------
    byte_string : str
        A string representation of a bytes literal, e.g. "b'\\x89PNG...'"

    Returns
    -------
    np.ndarray or None
        RGB image array, or None if decoding fails.
    """
    try:
        real_bytes = ast.literal_eval(byte_string)
        image_stream = io.BytesIO(real_bytes)
        with Image.open(image_stream) as img:
            img = img.convert('RGB')
            return np.array(img)
    except Exception:
        return None