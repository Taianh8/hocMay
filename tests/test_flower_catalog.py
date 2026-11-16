"""Tên lớp đúng thứ tự và ảnh tham khảo chỉ đến từ train."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import pytest
from src.flower_catalog import display_name, load_catalog, load_reference_index, reference_images
from src.inference import default_manifest_path, load_frozen_model, predict_image, predict_with_features
from src.utils import ROOT


def test_catalog_id_mapping():
    labels=load_catalog()
    assert len(labels)==102 and labels[1]['name_en']=='pink primrose'
    assert labels[77]['name_en']=='passion flower'
    assert labels[92]['name_en']=='bee balm' and display_name(labels[92])=='Hoa bạc hà ong'
    assert labels[102]['name_en']=='blackberry lily'


def test_flower_names_json_structure():
    import json
    p = ROOT / 'configs/flower_names.json'
    assert p.is_file()
    data = json.loads(p.read_text())
    assert len(data) == 102
    for i in range(1, 103):
        assert str(i) in data
        assert isinstance(data[str(i)]['name_en'], str) and len(data[str(i)]['name_en']) > 0


def test_top_k_predictions():
    from src.inference import predict_top_k
    manifest, payload = load_frozen_model(default_manifest_path())
    data = (ROOT / 'data/raw/jpg/image_06765.jpg').read_bytes()
    top_classes, projected = predict_top_k(data, manifest, payload, k=3)
    assert 1 <= len(top_classes) <= 3
    # Classes must be distinct
    c_ids = [c['class_id'] for c in top_classes]
    assert len(set(c_ids)) == len(c_ids)
    for c in top_classes:
        assert 1 <= c['class_id'] <= 102
        assert 0 < c['vote_share'] <= 1.0
    # Top 1 must match predict_image
    assert c_ids[0] == predict_image(data, manifest, payload)


def test_references_are_training_images_of_predicted_class():
    manifest, payload = load_frozen_model(default_manifest_path())
    data=(ROOT/'data/raw/jpg/image_06765.jpg').read_bytes()
    class_id, projected = predict_with_features(data, manifest, payload)
    assert class_id == predict_image(data, manifest, payload)
    index = load_reference_index(manifest, payload)
    refs = reference_images(projected, class_id, payload, index)
    assert len(refs)==3 and len(set(refs))==3
    train=set(loadmat(ROOT/'data/raw/setid.mat')['trnid'].ravel().tolist())
    labels=loadmat(ROOT/'data/raw/imagelabels.mat')['labels'].ravel()
    for path in refs:
        image_id=int(path.stem.split('_')[-1])
        assert image_id in train and labels[image_id-1]==class_id


def test_reference_mismatch_rejected():
    manifest, payload = load_frozen_model(default_manifest_path())
    manifest=dict(manifest);manifest['sha256']=dict(manifest['sha256'])
    manifest['sha256']['data/processed/features_v2.npz']='0'*64
    with pytest.raises(ValueError, match='tham khảo'):
        load_reference_index(manifest,payload)


def test_catalog_rejects_wrong_official_name(tmp_path, monkeypatch):
    import json
    import src.flower_catalog as catalog_module
    data=json.loads((ROOT/'configs/flower_names.json').read_text())
    data['92']['name_en']='wrong flower'
    path=tmp_path/'flower_names.json'; path.write_text(json.dumps(data))
    monkeypatch.setattr(catalog_module,'CATALOG',path)
    import pytest
    with pytest.raises(ValueError, match='không khớp'):
        catalog_module.load_catalog()
