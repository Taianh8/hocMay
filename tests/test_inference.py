"""Integration checks against the frozen, trusted V2 artifact and validation features."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.inference import default_manifest_path, image_features, load_frozen_model, predict_image
from src.utils import ROOT


def manifest_path():
    """Dùng cùng quy tắc chọn manifest như ứng dụng."""
    return default_manifest_path()


def load_current():
    return load_frozen_model(manifest_path())


def test_frozen_model_load_and_training_dimensions():
    manifest, payload = load_current()
    assert manifest['experiment'] == 'E6' and payload['pca'].n_samples_ == 1020
    assert payload['model'].n_features_in_ == 128
    assert np.array_equal(payload['model'].classes_, np.arange(102))


def test_image_features_and_prediction_match_saved_validation():
    manifest, payload = load_current()
    with np.load(ROOT/'data/processed/features_v2.npz', allow_pickle=False) as cache:
        idx = np.flatnonzero(cache['splits']=='val')[0]
        image_id = int(cache['ids'][idx]); expected = cache['h8_hsv'][idx]
        data = (ROOT/'data/raw/jpg'/f'image_{image_id:05d}.jpg').read_bytes()
        vector = image_features(data, manifest)
        assert vector.shape == (1812,) and np.array_equal(vector, expected)
    saved = pd.read_csv(ROOT/'results/v2/metrics/E6_validation_predictions.csv').set_index('image_id')
    predicted = predict_image(data, manifest, payload)
    assert isinstance(predicted, int) and 1 <= predicted <= 102
    assert predicted == int(saved.loc[image_id, 'predicted_class_id'])


def test_modified_artifact_hash_rejected(tmp_path):
    original = json.loads(manifest_path().read_text())
    original['sha256'][original['model_artifact']] = '0'*64
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='artifact đóng băng đã bị thay đổi'): load_frozen_model(path)


def test_invalid_input_rejected():
    manifest, _ = load_current()
    with pytest.raises(ValueError, match='Không thể giải mã ảnh'): image_features(b'not an image', manifest)


def test_frozen_model_matches_only_official_training_features():
    manifest, payload = load_current()
    with np.load(ROOT/'data/processed/features_v2.npz', allow_pickle=False) as cache:
        train = cache['splits'] == 'train'
        x, y = cache['h8_hsv'][train], cache['labels'][train]
        assert len(x) == 1020
        expected = payload['pca'].transform(x)
        assert np.allclose(expected, payload['model']._fit_X, atol=1e-6, rtol=1e-6)
        assert np.array_equal(y, payload['model']._y)


def test_streamlit_upload_flow(monkeypatch):
    from io import BytesIO
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    # Official train image: exercise the UI branch without making a test prediction.
    image = BytesIO((ROOT/'data/raw/jpg/image_06765.jpg').read_bytes())
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: image)
    app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    manifest, payload = load_current()
    expected = predict_image(image.getvalue(), manifest, payload)
    from src.flower_catalog import load_catalog, display_name
    result_cards = [element.value for element in app.markdown if '<div class="result-card">' in element.value]
    assert len(result_cards) == 1
    assert display_name(load_catalog()[expected]) in result_cards[0]
    assert f'{expected:02d}' in result_cards[0]
    assert any('ĐỐI CHIẾU BẰNG MẮT' in element.value for element in app.markdown)


def test_modified_manifest_preprocessing_rejected(tmp_path):
    manifest = json.loads(manifest_path().read_text())
    manifest['features']['histogram_bins_per_channel'] = 8
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Tiền xử lý hoặc đặc trưng'): load_frozen_model(path)
