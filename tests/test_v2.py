"""Tests for V2 feature behavior and training-only PCA fitting."""
import numpy as np
import pytest
from src.features_v2 import color_histogram, hog_color
from src import pipeline_v2


def test_color_histogram_and_hog_dimensions():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[..., 0] = 1.0
    rgb = color_histogram(image, 'rgb', 16)
    hsv = color_histogram(image, 'hsv', 16)
    assert rgb.shape == hsv.shape == (48,)
    assert np.linalg.norm(rgb) == pytest.approx(1)
    assert np.linalg.norm(hsv) == pytest.approx(1)
    assert not np.array_equal(rgb, hsv)
    h = {'orientations': 9, 'pixels_per_cell': 8, 'cells_per_block': 2}
    assert hog_color(image, h, 'rgb', 16).shape == (1812,)
    h['pixels_per_cell'] = 16
    assert hog_color(image, h, 'rgb', 16).shape == (372,)


def test_pca_fit_never_sees_validation(monkeypatch):
    calls = []
    class FakePCA:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def fit(self, x): calls.append(('fit', x.copy())); return self
        def transform(self, x): calls.append(('transform', x.copy())); return x[:, :2]
    monkeypatch.setattr(pipeline_v2, 'PCA', FakePCA)
    train = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    val = np.array([[100, 101, 102]], dtype=float)
    cfg = {'pca_max_components': 2, 'pca_iterated_power': 3, 'seed': 42}
    _, train_out, val_out, _, _ = pipeline_v2.fit_projection(train, val, cfg)
    assert len(calls) == 3 and calls[0][0] == 'fit' and np.array_equal(calls[0][1], train)
    assert calls[1][0] == 'transform' and np.array_equal(calls[1][1], train)
    assert calls[2][0] == 'transform' and np.array_equal(calls[2][1], val)
    assert train_out.shape == (2, 2) and val_out.shape == (1, 2)


def test_unknown_color_space_rejected():
    with pytest.raises(ValueError, match='Unknown color space'):
        color_histogram(np.zeros((4, 4, 3)), 'lab', 16)
