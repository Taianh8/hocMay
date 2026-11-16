"""Tests for V5 experiments and leak-free fitting."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.utils import ROOT
from src.pipeline_v5 import load_config, run_e7_grid


def test_v5_config_and_unified_comparison():
    cfg = load_config()
    assert 'decision_tree' in cfg
    assert set(cfg['decision_tree']['criterion']) == {'gini', 'entropy'}
    assert set(cfg['decision_tree']['max_depth']) == {10, 20, None}
    assert set(cfg['decision_tree']['min_samples_leaf']) == {1, 5}

    comp_file = ROOT / 'results/v5/unified_comparison.json'
    assert comp_file.is_file(), "unified_comparison.json should exist"
    data = json.loads(comp_file.read_text())
    assert data['selected_model'] == 'E6'
    assert 'validation macro F1' in data['selection_rule']
    assert len(data['unified_comparison']) >= 7


def test_v5_pca_fit_strictly_on_train_mock(monkeypatch):
    calls = []
    import src.pipeline_v5 as p5

    class MockPCA:
        def __init__(self, **kwargs):
            pass
        def fit(self, X):
            calls.append(('fit', X.shape))
            return self
        def transform(self, X):
            calls.append(('transform', X.shape))
            return np.zeros((X.shape[0], 2))

    monkeypatch.setattr(p5, 'PCA', MockPCA)
    X_tr = np.zeros((10, 20))
    y_tr = np.zeros(10, dtype=int)
    X_va = np.zeros((5, 20))
    y_val = np.zeros(5, dtype=int)
    cfg = {
        'seed': 42,
        'pca_components': 2,
        'pca_iterated_power': 3,
        'decision_tree': {
            'random_state': 42,
            'criterion': ['gini'],
            'max_depth': [10],
            'min_samples_leaf': [1]
        }
    }
    p5.run_e7_grid(X_tr, y_tr, X_va, y_val, cfg)
    # Ensure fit was called strictly on X_tr (shape (10, 20)), never on val
    assert calls[0] == ('fit', (10, 20))
    assert calls[1] == ('transform', (10, 20))
    assert calls[2] == ('transform', (5, 20))
