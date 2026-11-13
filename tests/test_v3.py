"""V3's split boundaries and fold-local learned transforms."""
import numpy as np
import pytest
from src.pipeline_v3 import make_folds, run_fold
from src import pipeline_v3


def test_cv_partitions_official_training_ids_only():
    # Ten training images per class; unrelated IDs stand for official validation.
    labels = np.repeat(np.arange(102), 10)
    ids = np.arange(1, 1021)
    val_ids = set(range(1021, 2041))
    folds = make_folds(labels, ids, 5, 42)
    assert len(folds) == 5
    seen = []
    for fold in folds:
        train, held = set(fold['train_image_ids']), set(fold['holdout_image_ids'])
        assert len(train) == 816 and len(held) == 204
        assert not (train & held) and not ((train | held) & val_ids)
        assert train | held == set(ids)
        seen.extend(held)
    assert sorted(seen) == list(ids)
    assert folds == make_folds(labels, ids, 5, 42)


def test_pca_fit_only_cv_training_fold(monkeypatch):
    calls = []
    class FakePCA:
        def __init__(self, **kwargs): pass
        def fit(self, x): calls.append(('fit', x.copy())); return self
        def transform(self, x): calls.append(('transform', x.copy())); return x
    class FakeKNN:
        def __init__(self, **kwargs): pass
        def fit(self, x, y): calls.append(('knn_fit', x.copy())); return self
        def predict(self, x): calls.append(('predict', x.copy())); return np.zeros(len(x), dtype=int)
    monkeypatch.setattr(pipeline_v3, 'PCA', FakePCA)
    monkeypatch.setattr(pipeline_v3, 'KNeighborsClassifier', FakeKNN)
    x = np.array([[1.,2.],[3.,4.],[999.,998.]])
    model = {'pca_components': 2, 'pca_iterated_power': 3, 'params': {'n_neighbors': 1}}
    run_fold(x, np.zeros(3, dtype=int), np.array([0,1]), np.array([2]), model, 42)
    assert np.array_equal(calls[0][1], x[:2]) and calls[0][0] == 'fit'
    assert np.array_equal(calls[1][1], x[:2]) and calls[1][0] == 'transform'
    assert np.array_equal(calls[2][1], x[2:]) and calls[2][0] == 'transform'
    assert np.array_equal(calls[3][1], x[:2]) and calls[3][0] == 'knn_fit'
    assert np.array_equal(calls[4][1], x[2:]) and calls[4][0] == 'predict'
