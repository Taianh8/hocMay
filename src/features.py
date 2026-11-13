"""Deterministic pixel and grayscale HOG features."""
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

def extract(image: np.ndarray, kind: str, settings: dict) -> np.ndarray:
    """Extract a fixed-length vector from a normalized RGB image."""
    if kind == 'pixels':
        return image.ravel().astype(np.float32)
    if kind == 'hog':
        h = settings['hog']
        return hog(rgb2gray(image), orientations=h['orientations'], pixels_per_cell=(h['pixels_per_cell'],)*2, cells_per_block=(h['cells_per_block'],)*2, block_norm='L2-Hys').astype(np.float32)
    raise ValueError(f'Unknown feature type: {kind}')
