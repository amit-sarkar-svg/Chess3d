"""
src/utils/stb_image.py

Lightweight stb_image-like wrapper for loading images in Python.

Provides:
    load(path, flip_vertically=True)
Returns:
    (data, width, height, channels)
Where:
    data is a raw bytes buffer suitable for OpenGL texture upload.

This wrapper imitates stb_image behavior.

Dependencies:
    Pillow (PIL)
"""

from __future__ import annotations
from PIL import Image
import numpy as np
import os


def load(path: str, flip_vertically: bool = True):
    """
    Load an image file and return raw RGBA bytes + dimensions.

    Parameters:
        path (str) : path to texture file
        flip_vertically (bool): flip for OpenGL UV origin

    Returns:
        (data, width, height, channels)

    Raises:
        FileNotFoundError if path does not exist
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[stb_image] File not found: {path}")

    img = Image.open(path)

    # Convert to RGBA for consistent GPU uploads
    img = img.convert("RGBA")

    if flip_vertically:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    width, height = img.size
    data = img.tobytes("raw", "RGBA", 0, -1)

    return data, width, height, 4


def load_numpy(path: str, flip_vertically: bool = True):
    """
    Load image and return a numpy array instead of raw bytes.
    Useful for CPU image processing.

    Returns:
        img_array (H x W x 4 uint8 numpy array)
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"[stb_image] File not found: {path}")

    img = Image.open(path).convert("RGBA")
    if flip_vertically:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    arr = np.array(img, dtype=np.uint8)
    return arr
