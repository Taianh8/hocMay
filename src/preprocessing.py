"""Deterministic image conversion; no statistics are learned."""
from pathlib import Path
import numpy as np
from PIL import Image

def preprocess(path: Path, size: int) -> np.ndarray:
    """Convert to RGB, resize square with bilinear interpolation and normalize to [0, 1]."""
    with Image.open(path) as image:
        image = image.convert('RGB').resize ( (size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0
