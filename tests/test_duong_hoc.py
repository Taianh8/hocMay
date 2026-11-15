"""Các bất biến chia mẫu và tính chỉ số của đường học."""
import numpy as np
from scripts.duong_hoc import subsets, score


def test_subsets_are_stratified_nested_and_reproducible():
    y = np.repeat(np.arange(102), 10)
    train = np.concatenate([np.arange(i*10, i*10+8) for i in range(102)])
    holdout = set(range(1020)) - set(train)
    chosen = subsets(y, train, 1)
    for n, idx in chosen.items():
        assert len(idx) == 102*n and not (set(idx) & holdout)
        assert np.array_equal(np.bincount(y[idx], minlength=102), np.full(102,n))
        assert np.array_equal(idx, subsets(y, train, 1)[n])
    assert set(chosen[2]) <= set(chosen[4]) <= set(chosen[6]) <= set(chosen[8])


def test_score_uses_all_102_classes():
    accuracy, f1 = score(np.array([0, 0]), np.array([0, 0]))
    assert accuracy == 1.0
    assert np.isclose(f1, 1/102)
