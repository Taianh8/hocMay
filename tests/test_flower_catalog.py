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
