"""Xác nhận đánh giá validation không sử dụng mẫu test."""
import numpy as np
import pandas as pd
from src import validation


def test_evaluate_validation_filters_official_split(tmp_path, monkeypatch):
    x = np.array([[1.0], [99.0], [2.0], [999.0]])
    features = {'splits': np.array(['train','val','val','test']), 'pixels': x,
                'labels': np.array([0, 1, 2, 3]), 'ids': np.array([1, 2, 3, 4])}
    class FakeModel:
        def predict(self, data):
            assert np.array_equal(data, np.array([[99.0], [2.0]]))
            return np.array([1, 2])
    monkeypatch.setattr(validation.p, 'load_features', lambda df: features)
    monkeypatch.setattr(validation.p, 'selected', lambda cfg: {'experiment':'E1','model':FakeModel()})
    monkeypatch.setattr(validation.p, 'MET', tmp_path)
    monkeypatch.setattr(validation.p, 'FIG', tmp_path)
    monkeypatch.setattr(validation, 'plot_confusion', lambda *a, **kw: np.zeros((102,102),dtype=int))
    result=validation.evaluate_validation(pd.DataFrame(), {'experiments':{'E1':{'features':'pixels'}}})
    saved=pd.read_csv(tmp_path/'validation_predictions.csv')
    assert result['split']=='val' and result['metrics']['accuracy']==1.0
    assert saved.image_id.tolist()==[2,3]
    assert saved.true_class_id.tolist()==[2,3]


def test_validation_errors_rejects_test_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(validation.p, 'MET', tmp_path)
    pd.DataFrame({'image_id':[9], 'true_class_id':[1], 'predicted_class_id':[2]}).to_csv(tmp_path/'validation_predictions.csv', index=False)
    df=pd.DataFrame({'image_id':[2,9], 'class_id':[1,1], 'split':['val','test'], 'path':['val.jpg','test.jpg']})
    import pytest
    with pytest.raises(ValueError, match='không khớp'):
        validation.validation_errors(df)
