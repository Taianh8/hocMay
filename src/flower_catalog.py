"""Tên và ảnh tham khảo cho các lớp Oxford 102 Flowers."""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .utils import ROOT, digest

CATALOG = ROOT/'configs/flowers102_labels.json'


def load_catalog() -> dict[int, dict]:
    """Đọc đúng 102 tên lớp, dùng ID Oxford 1–102."""
    data = json.loads(CATALOG.read_text())['labels']
    if len(data) != 102 or [x['class_id'] for x in data] != list(range(1, 103)):
        raise ValueError('Danh sách tên hoa không đủ 102 lớp theo thứ tự.')
    if any(not isinstance(x.get('name_en'), str) or not x['name_en'] for x in data):
        raise ValueError('Danh sách tên hoa chứa tên không hợp lệ.')
    return {row['class_id']: row for row in data}


def display_name(item: dict) -> str:
    """Ưu tiên tên Việt đã bổ sung; giữ tên gốc khi chưa chắc chắn."""
    return item.get('name_vi') or item['name_en'].capitalize()


def load_reference_index(manifest: dict, payload: dict) -> dict:
    """Xác minh ảnh tham khảo thuộc train và căn chỉnh với model đã fit."""
    paths = ('data/processed/features_v2.npz', 'data/raw/imagelabels.mat', 'data/raw/setid.mat')
    for path in paths:
        file = ROOT/path
        if not file.is_file() or digest(file) != manifest['sha256'].get(path):
            raise ValueError('Chưa có dữ liệu tham khảo khớp với model. Hãy chuẩn bị lại theo README.')
    labels = loadmat(ROOT/paths[1])['labels'].ravel().astype(int)-1
    official_train = np.sort(loadmat(ROOT/paths[2])['trnid'].ravel())
    with np.load(ROOT/paths[0], allow_pickle=False) as f:
        mask = f['splits']=='train'; ids = f['ids'][mask]; y = f['labels'][mask]; X = f['h8_hsv'][mask]
    model, pca = payload['model'], payload['pca']
    if not np.array_equal(ids, official_train) or not np.array_equal(labels[ids-1],y):
        raise ValueError('ID hoặc nhãn ảnh tham khảo không khớp split train.')
    if not np.array_equal(model._y, y) or not np.allclose(model._fit_X, pca.transform(X), atol=1e-6, rtol=1e-6):
        raise ValueError('Ảnh tham khảo không khớp model được huấn luyện.')
    return {'ids':ids, 'labels':y}


def reference_images(projected: np.ndarray, class_id: int, payload: dict, index: dict, limit: int = 3) -> list[Path]:
    """Lấy tối đa ba ảnh train gần nhất thuộc lớp đã dự đoán."""
    model = payload['model']
    _, order = model.kneighbors(projected, n_neighbors=model.n_samples_fit_)
    result = []
    for i in order[0]:
        if int(index['labels'][i]) == class_id-1:
            path = ROOT/'data/raw/jpg'/f'image_{int(index["ids"][i]):05d}.jpg'
            if path.is_file(): result.append(path)
            if len(result) == limit: break
    return result
