"""Fixed, per-image HOG and color features for V2."""
import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog


def unit(vector: np.ndarray) -> np.ndarray:
    """L2-normalize an individual feature block; preserve all-zero vectors."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def color_histogram(image: np.ndarray, space: str, bins: int) -> np.ndarray:
    """Concatenate per-channel normalized histograms in RGB or HSV."""
    if space == 'rgb':
        color = image
    elif space == 'hsv':
        color = rgb2hsv(image)
    else:
        raise ValueError(f'Unknown color space: {space}')
    hist = [np.histogram(color[..., c], bins=bins, range=(0, 1))[0].astype(np.float32) for c in range(3)]
    return unit(np.concatenate(hist))


def hog_feature(image: np.ndarray, hog_settings: dict) -> np.ndarray:
    """Return an independently normalized grayscale HOG block."""
    h = hog_settings
    edge = hog(rgb2gray(image), orientations=h['orientations'], pixels_per_cell=(h['pixels_per_cell'],)*2, cells_per_block=(h['cells_per_block'],)*2, block_norm='L2-Hys')
    return unit(edge)


def hog_color(image: np.ndarray, hog_settings: dict, space: str, bins: int) -> np.ndarray:
    """Concatenate unit-normalized HOG and color blocks with equal block weight."""
    return np.concatenate((hog_feature(image, hog_settings), color_histogram(image, space, bins))).astype(np.float32)
