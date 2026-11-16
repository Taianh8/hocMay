"""Bảo vệ phép chọn validation và chống chạy lại test cuối."""
import pytest
from src.pipeline_v5 import select_best
import scripts.run_final_test as final_test


def test_selection_includes_control_and_accuracy_tiebreak():
    rows=[{'experiment':'E6','val_macro_f1':0.212,'val_accuracy':0.23},
          {'experiment':'E6_bins32','val_macro_f1':0.215,'val_accuracy':0.23},
          {'experiment':'E7','val_macro_f1':0.215,'val_accuracy':0.24}]
    assert select_best(rows)['experiment']=='E7'
    assert select_best(rows[:2])['experiment']=='E6_bins32'
    with pytest.raises(ValueError): select_best([])


def test_existing_final_output_prevents_second_evaluation(tmp_path,monkeypatch):
    out=tmp_path/'final_test'; out.mkdir(); (out/'metrics.json').write_text('{}')
    monkeypatch.setattr(final_test,'OUT',out)
    monkeypatch.setattr(final_test,'ROOT',tmp_path)
    monkeypatch.setattr(final_test,'load_dataset',lambda: pytest.fail('Không được đọc dữ liệu test'))
    with pytest.raises(RuntimeError, match='từ chối chạy lại'):
        final_test.main()


def test_weighted_vote_not_nearest_neighbor(monkeypatch):
    from sklearn.neighbors import KNeighborsClassifier
    import src.inference as inference
    class Identity:
        def transform(self,x): return x
    model=KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance').fit([[0.1],[0.15],[-0.15]],[0,1,1])
    monkeypatch.setattr(inference,'image_features',lambda data,manifest: __import__('numpy').array([0.]))
    top,_=inference.predict_top_k(b'x',{'pca':{'components':1}},{'pca':Identity(),'model':model})
    assert top[0]['class_id']==2 and top[1]['class_id']==1
    assert top[0]['vote_share'] > top[1]['vote_share']


def test_cv_uses_e7_validation_winner_and_fits_pca_inside_fold(monkeypatch):
    import numpy as np
    import src.pipeline_v5 as v5
    calls=[]
    class OneFold:
        def __init__(self,**kwargs): pass
        def split(self,x,y): yield np.array([0,1,2]),np.array([3])
    class FakePCA:
        def __init__(self,**kwargs): pass
        def fit(self,x): calls.append(('pca_fit',x.copy()));return self
        def transform(self,x): return x
    class FakeTree:
        def __init__(self,**kwargs): calls.append(('tree_params',kwargs))
        def fit(self,x,y): return self
        def predict(self,x): return np.zeros(len(x),dtype=int)
    monkeypatch.setattr(v5,'StratifiedKFold',OneFold)
    monkeypatch.setattr(v5,'PCA',FakePCA)
    monkeypatch.setattr(v5,'DecisionTreeClassifier',FakeTree)
    x=np.array([[1.],[2.],[3.],[999.]])
    cfg={'pca_components':1,'pca_iterated_power':3,'seed':42,'decision_tree':{'random_state':42}}
    winner={'feature':'h8_hsv_pca','params':{'criterion':'entropy','max_depth':10,'min_samples_leaf':5}}
    v5.run_5fold_cv(x,np.zeros(4,dtype=int),'E7',winner,cfg)
    assert np.array_equal(calls[0][1],x[:3])
    assert calls[1][1]['criterion']=='entropy' and calls[1][1]['min_samples_leaf']==5
