"""Fixtures for checks that depend on locally generated artifacts."""
import json

import pytest

from src.inference import default_manifest_path
from src.utils import ROOT


@pytest.fixture
def integration_artifacts():
    manifest_path = default_manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    paths = [ROOT / manifest['model_artifact'], ROOT / 'data/processed/features_v2.npz',
             ROOT / 'results/v2/metrics/E6_validation_predictions.csv',
             ROOT / 'data/raw/jpg/image_06765.jpg', ROOT / 'data/raw/imagelabels.mat',
             ROOT / 'data/raw/setid.mat']
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        pytest.skip('Thiếu artifact tích hợp: ' + ', '.join(missing) +
                    '. Tải dataset và chạy python scripts/chuan_bi_ui.py; xem README.')
