"""Đo đường học của E1/E4/E6 bằng năm fold có sẵn trên train chính thức.

Không sử dụng validation chính thức hoặc test; không thay đổi model/artifact cũ.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.features_v2 import hog_color
from src.pipeline_v3 import fixed_winners, load_config, official_metadata, validate_v2_cache
from src.preprocessing import preprocess
from src.utils import ROOT, digest, save_json

OUT = ROOT / 'results/duong_hoc'
FIG = ROOT / 'reports/latex/figures'
SIZES = (2, 4, 6, 8)
SEED = 42


def subsets(y: np.ndarray, fold_train: np.ndarray, fold_number: int) -> dict[int, np.ndarray]:
    """Chọn tập con phân tầng, lồng nhau; cùng các mẫu cho mọi model."""
    groups = []
    for label in range(102):
        group = fold_train[y[fold_train] == label]
        if len(group) != 8:
            raise ValueError(f'Fold {fold_number}: lớp {label+1} không có 8 ảnh train')
        groups.append(np.random.default_rng(SEED + fold_number * 1000 + label).permutation(group))
    return {n: np.sort(np.concatenate([group[:n] for group in groups])) for n in SIZES}


def score(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Tính Accuracy và F1 vĩ mô trên đủ 102 lớp."""
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, labels=range(102), average='macro', zero_division=0))


def run():
    """Tính điểm thực trên tập fit và holdout, lưu riêng với kết quả lịch sử."""
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); fixed_winners(cfg)
    labels, ids = official_metadata(); cached = validate_v2_cache(labels, ids)
    train_ids = np.sort(ids['train']); y = labels[train_ids-1]
    features, pixels = [], []
    settings = {'orientations': 9, 'pixels_per_cell': 8, 'cells_per_block': 2}
    for image_id in train_ids:
        image = preprocess(ROOT/'data/raw/jpg'/f'image_{image_id:05d}.jpg', 64)
        feature = hog_color(image, settings, 'hsv', 16)
        if not np.array_equal(feature, cached[int(image_id)]):
            raise ValueError(f'Ảnh train {image_id} không khớp cache V2 đã xác thực')
        pixels.append(image.ravel()); features.append(feature)
    arrays = {'pixels': np.stack(pixels), 'h8_hsv': np.stack(features)}
    folds = json.loads((ROOT/'results/v3/metrics/folds.json').read_text())
    if len(folds) != 5: raise ValueError('Cần đúng năm fold V3 đã lưu')
    rows = []; subset_ids = []; predictions = []
    for fold_number, fold in enumerate(folds, 1):
        train_idx = np.asarray(fold['train_index']); hold_idx = np.asarray(fold['holdout_index'])
        if not np.array_equal(train_ids[train_idx], fold['train_image_ids']) or not np.array_equal(train_ids[hold_idx], fold['holdout_image_ids']):
            raise ValueError('Fold V3 không khớp ID train chính thức')
        if set(train_idx) & set(hold_idx): raise ValueError('Fold train và holdout bị chồng lấn')
        subsets_by_size = subsets(y, train_idx, fold_number)
        for n, subset in subsets_by_size.items():
            subset_ids.append({'fold':fold_number, 'images_per_class':n, 'train_image_ids':train_ids[subset].tolist(), 'holdout_image_ids':train_ids[hold_idx].tolist()})
            for name in ('E1','E4','E6'):
                model_cfg = cfg['models'][name]
                X = arrays[model_cfg['feature']]; fit_x = X[subset]; held_x = X[hold_idx]
                if name == 'E6':
                    pca = PCA(n_components=128, svd_solver='randomized', iterated_power=3, random_state=SEED)
                    pca.fit(fit_x); fit_x = pca.transform(fit_x); held_x = pca.transform(held_x)
                model = KNeighborsClassifier(**model_cfg['params'], n_jobs=1).fit(fit_x, y[subset])
                train_prediction = model.predict(fit_x); holdout_prediction = model.predict(held_x)
                train_accuracy, train_f1 = score(y[subset], train_prediction)
                held_accuracy, held_f1 = score(y[hold_idx], holdout_prediction)
                predictions.append({'experiment':name, 'fold':fold_number, 'images_per_class':n, 'train_prediction':train_prediction.tolist(), 'holdout_prediction':holdout_prediction.tolist()})
                rows.append({'experiment':name, 'fold':fold_number, 'images_per_class':n, 'train_samples':len(subset), 'holdout_samples':len(hold_idx), 'train_accuracy':train_accuracy, 'train_macro_f1':train_f1, 'holdout_accuracy':held_accuracy, 'holdout_macro_f1':held_f1})
                print(f'{name}: fold {fold_number}, {len(subset)} ảnh fit, F1 holdout={held_f1:.4f}',flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(OUT/'ket_qua_theo_fold.csv', index=False)
    save_json(OUT/'fold_va_mau.json', subset_ids)
    save_json(OUT/'du_doan.json', predictions)
    summary = frame.groupby(['experiment','train_samples','images_per_class'],as_index=False).agg(train_f1=('train_macro_f1','mean'),holdout_f1=('holdout_macro_f1','mean'),holdout_f1_std=('holdout_macro_f1',lambda x: x.std(ddof=0)),train_accuracy=('train_accuracy','mean'),holdout_accuracy=('holdout_accuracy','mean'))
    summary.to_csv(OUT/'tong_hop.csv', index=False)
    files = ['configs/experiments_v3.yaml','results/v3/metrics/folds.json','data/raw/imagelabels.mat','data/raw/setid.mat','data/processed/features_v2.npz','src/preprocessing.py','src/features_v2.py','scripts/duong_hoc.py']
    h = hashlib.sha256()
    for image_id in train_ids: h.update(str(image_id).encode()); h.update(digest(ROOT/'data/raw/jpg'/f'image_{image_id:05d}.jpg').encode())
    save_json(OUT/'nguon_goc.json', {'scope':'official train only; five V3 folds; no official validation/test prediction','classes':102,'sizes_per_class':list(SIZES),'fold_seed':42,'subset_seed':'42 + 1000 * fold(1..5) + zero_based_class','pca_seed':SEED,'pca_components':128,'training_score':'in-sample predictions; includes each training sample as its own neighbor','dependencies':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','scikit-image','Pillow']},'sha256':{p:digest(ROOT/p) for p in files},'train_images_sha256':h.hexdigest()})
    fig, ax=plt.subplots(figsize=(8,4.5))
    train_curves=[summary[summary.experiment==name].train_f1.to_numpy() for name in ('E1','E4','E6')]
    if all(np.allclose(train_curves[0],curve) for curve in train_curves[1:]):
        part=summary[summary.experiment=='E1']; ax.plot(part.train_samples,part.train_f1,'k--',label='F1 tập fit (E1, E4, E6 trùng nhau)')
    else:
        for name in ('E1','E4','E6'):
            part=summary[summary.experiment==name]; ax.plot(part.train_samples,part.train_f1,'--',label=f'{name}: F1 tập fit')
    for name in ('E1','E4','E6'):
        part=summary[summary.experiment==name]
        ax.plot(part.train_samples,part.holdout_f1,'o-',label=f'{name}: F1 fold giữ lại')
        ax.fill_between(part.train_samples,part.holdout_f1-part.holdout_f1_std,part.holdout_f1+part.holdout_f1_std,alpha=.08)
    ax.set(title='Đường học trên train chính thức — 5 fold',xlabel='Số ảnh dùng để fit',ylabel='F1 trung bình vĩ mô',ylim=(0,1.08));ax.grid(alpha=.25);ax.legend(fontsize=9,loc='center right');fig.tight_layout();fig.savefig(FIG/'duong_hoc.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in ['E1','E4','E6']:
        part=summary[summary.experiment==name];ax.plot(part.train_samples,part.holdout_f1,'o-',label=name)
    ax.set(title='So sánh F1 trên cùng fold giữ lại',xlabel='Số ảnh dùng để fit',ylabel='F1 trung bình vĩ mô');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/'so_sanh_duong_hoc.png',dpi=150);plt.close(fig)
    print('Đã ghi kết quả 60 lần fit; không dự đoán trên validation chính thức hoặc test.')

if __name__ == '__main__': run()
