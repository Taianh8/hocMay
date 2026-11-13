"""Critical data and leakage invariants with small synthetic inputs."""
import numpy as np
import pytest
from scipy.io import savemat
from PIL import Image
from src import data_loader
from src.features import extract
from src.preprocessing import preprocess
from src.models import search

def make_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, 'EXPECTED', 102)
    (tmp_path / 'jpg').mkdir()
    savemat(tmp_path / 'imagelabels.mat', {'labels': np.arange(1, 103)})
    savemat(tmp_path / 'setid.mat', {'trnid': np.arange(1, 35), 'valid': np.arange(35, 69), 'tstid': np.arange(69, 103)})
    for i in range(1, 103): Image.new('RGB', (4, 5), (i, 0, 0)).save(tmp_path / 'jpg' / f'image_{i:05d}.jpg')
    return tmp_path

def test_mapping_and_disjoint_splits(tmp_path, monkeypatch):
    df = data_loader.load_dataset(make_dataset(tmp_path, monkeypatch))
    assert len(df) == 102
    assert (df.label == df.class_id - 1).all()
    assert df.image_id.is_unique

def test_overlap_rejected(tmp_path, monkeypatch):
    make_dataset(tmp_path, monkeypatch)
    savemat(tmp_path / 'setid.mat', {'trnid': [1, 2], 'valid': [2], 'tstid': np.arange(3, 103)})
    with pytest.raises(ValueError, match='disjoint'): data_loader.load_dataset(tmp_path)

def test_corrupt_image_rejected(tmp_path, monkeypatch):
    make_dataset(tmp_path, monkeypatch)
    (tmp_path / 'jpg/image_00001.jpg').write_bytes(b'bad')
    with pytest.raises(ValueError, match='invalid images'): data_loader.load_dataset(tmp_path)

def test_features_and_preprocessing(tmp_path):
    path = tmp_path / 'test.png'; Image.new('RGB', (20, 30), (255, 0, 0)).save(path)
    image = preprocess(path, 64)
    cfg = {'hog': {'orientations': 9, 'pixels_per_cell': 8, 'cells_per_block': 2}}
    assert image.shape == (64, 64, 3) and image.min() == 0 and image.max() == 1
    assert extract(image, 'pixels', cfg).shape == (12288,)
    assert extract(image, 'hog', cfg).shape == (1764,)

def test_search_fits_only_train(monkeypatch):
    calls = []
    class Fake:
        def fit(self, x, y): calls.append((x.copy(), y.copy())); return self
        def predict(self, x): return np.zeros(len(x), dtype=int)
    monkeypatch.setattr('src.models.candidates', lambda spec, seed: [({}, Fake())])
    train = np.array([[1], [2]]); val = np.array([[99]])
    search(train, np.array([0, 0]), val, np.array([0]), {}, 42)
    assert len(calls) == 1 and np.array_equal(calls[0][0], train)
