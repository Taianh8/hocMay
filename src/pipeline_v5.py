"""Pipeline V5: Thí nghiệm bổ sung E7 (Cây quyết định công bằng) và Thí nghiệm đối chứng.

Chỉ sử dụng tập huấn luyện (train) và kiểm định (validation/CV). Tuyệt đối không dùng test.
Mọi phép biến đổi học tham số (PCA) chỉ fit trên tập train hoặc từng fold huấn luyện trong CV.
"""
from itertools import product
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .data_loader import load_dataset
from .features_v2 import hog_color
from .preprocessing import preprocess
from .utils import ROOT, digest, save_json

OUT = ROOT / 'results/v5'
CFG_PATH = ROOT / 'configs/experiments_v5.yaml'
CACHE_PATH = OUT / 'features_v5.npz'


def load_config():
    with CFG_PATH.open() as f:
        return yaml.safe_load(f)


def extract_or_load_features(df, cfg):
    """Extract or load cached features for train and val splits."""
    OUT.mkdir(parents=True, exist_ok=True)
    selected = df[df.split.isin(['train', 'val'])].reset_index(drop=True)
    
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        if (np.array_equal(arrays['ids'], selected.image_id.to_numpy()) and
            np.array_equal(arrays['labels'], selected.label.to_numpy()) and
            np.array_equal(arrays['splits'], selected.split.to_numpy())):
            return selected, arrays

    extracted = {'E6_size128': [], 'E6_bins32': [], 'h8_hsv': []}
    h8_settings = {'orientations': 9, 'pixels_per_cell': 8, 'cells_per_block': 2}
    
    for row in selected.itertuples():
        p = Path(row.path)
        # E6_size128: size 128, bins 16
        im128 = preprocess(p, 128)
        extracted['E6_size128'].append(hog_color(im128, h8_settings, 'hsv', 16))
        
        # E6_bins32: size 64, bins 32
        im64 = preprocess(p, 64)
        extracted['E6_bins32'].append(hog_color(im64, h8_settings, 'hsv', 32))
        
        # h8_hsv: size 64, bins 16
        extracted['h8_hsv'].append(hog_color(im64, h8_settings, 'hsv', 16))

    arrays = {k: np.stack(v).astype(np.float32) for k, v in extracted.items()}
    arrays.update(
        ids=selected.image_id.to_numpy(),
        labels=selected.label.to_numpy(),
        splits=selected.split.to_numpy(dtype=str)
    )
    np.savez_compressed(CACHE_PATH, **arrays)
    return selected, arrays


def run_e7_grid(X_train_raw, y_train, X_val_raw, y_val, cfg):
    """Grid search DecisionTreeClassifier on h8_hsv (raw and with PCA)."""
    # Fit PCA on train only
    pca = PCA(n_components=cfg['pca_components'], svd_solver='randomized',
              iterated_power=cfg['pca_iterated_power'], random_state=cfg['seed'])
    t0 = perf_counter()
    pca.fit(X_train_raw)
    pca_fit_sec = perf_counter() - t0
    X_train_pca = pca.transform(X_train_raw)
    X_val_pca = pca.transform(X_val_raw)

    dt_cfg = cfg['decision_tree']
    grid = list(product(
        [('h8_hsv', X_train_raw, X_val_raw, 0.0),
         ('h8_hsv_pca', X_train_pca, X_val_pca, pca_fit_sec)],
        dt_cfg['criterion'],
        dt_cfg['max_depth'],
        dt_cfg['min_samples_leaf']
    ))

    results = []
    for (feat_name, X_tr, X_va, pca_sec), crit, depth, leaf in grid:
        clf = DecisionTreeClassifier(criterion=crit, max_depth=depth,
                                     min_samples_leaf=leaf, random_state=dt_cfg['random_state'])
        t0 = perf_counter()
        clf.fit(X_tr, y_train)
        fit_sec = perf_counter() - t0
        
        t0 = perf_counter()
        pred = clf.predict(X_va)
        pred_sec = perf_counter() - t0

        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred, average='macro', zero_division=0))
        
        results.append({
            'experiment': 'E7',
            'feature': feat_name,
            'params': {'criterion': crit, 'max_depth': depth, 'min_samples_leaf': leaf},
            'accuracy': acc,
            'macro_f1': f1,
            'fit_seconds': fit_sec,
            'predict_seconds': pred_sec,
            'pca_fit_seconds': pca_sec
        })

    return results


def run_controlled_val(feature_arrays, y_train, y_val, splits, cfg):
    """Run E6_size128 and E6_bins32 on validation."""
    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    results = []

    for name, exp_cfg in cfg['controlled_experiments'].items():
        X_tr_raw = feature_arrays[name][train_mask]
        X_va_raw = feature_arrays[name][val_mask]

        pca = PCA(n_components=exp_cfg['pca_components'], svd_solver='randomized',
                  iterated_power=cfg['pca_iterated_power'], random_state=cfg['seed'])
        t0 = perf_counter()
        pca.fit(X_tr_raw)
        pca_sec = perf_counter() - t0
        X_tr = pca.transform(X_tr_raw)
        X_va = pca.transform(X_va_raw)

        knn_cfg = exp_cfg['knn']
        knn = KNeighborsClassifier(**knn_cfg)
        t0 = perf_counter()
        knn.fit(X_tr, y_train)
        fit_sec = perf_counter() - t0
        
        t0 = perf_counter()
        pred = knn.predict(X_va)
        pred_sec = perf_counter() - t0

        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred, average='macro', zero_division=0))

        results.append({
            'experiment': name,
            'feature': name,
            'params': knn_cfg,
            'accuracy': acc,
            'macro_f1': f1,
            'fit_seconds': fit_sec,
            'predict_seconds': pred_sec,
            'pca_fit_seconds': pca_sec
        })

    return results


def run_5fold_cv(X_train_raw, y_train, exp_name, exp_cfg, cfg, fold_seed=42):
    """5-fold Stratified CV on training set only (fold_seed=42)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    cv_records = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train), start=1):
        X_tr_fold = X_train_raw[tr_idx]
        y_tr_fold = y_train[tr_idx]
        X_va_fold = X_train_raw[val_idx]
        y_va_fold = y_train[val_idx]

        if exp_name == 'E7':
            if not exp_cfg or exp_cfg['feature'] not in ('h8_hsv', 'h8_hsv_pca'):
                raise ValueError('Cần cấu hình E7 thắng trên validation')
            pca_sec = 0.0
            if exp_cfg['feature'] == 'h8_hsv_pca':
                pca = PCA(n_components=cfg['pca_components'], svd_solver='randomized',
                          iterated_power=cfg['pca_iterated_power'], random_state=cfg['seed'])
                t0 = perf_counter(); pca.fit(X_tr_fold); pca_sec = perf_counter() - t0
                X_tr_fold = pca.transform(X_tr_fold); X_va_fold = pca.transform(X_va_fold)
            clf = DecisionTreeClassifier(**exp_cfg['params'], random_state=cfg['decision_tree']['random_state'])
            t0 = perf_counter(); clf.fit(X_tr_fold, y_tr_fold); fit_sec = perf_counter() - t0
            pred = clf.predict(X_va_fold)
        else:
            pca = PCA(n_components=exp_cfg['pca_components'], svd_solver='randomized',
                      iterated_power=cfg['pca_iterated_power'], random_state=cfg['seed'])
            t0 = perf_counter()
            pca.fit(X_tr_fold)
            pca_sec = perf_counter() - t0
            X_tr_pca = pca.transform(X_tr_fold)
            X_va_pca = pca.transform(X_va_fold)

            knn = KNeighborsClassifier(**exp_cfg['knn'])
            t0 = perf_counter()
            knn.fit(X_tr_pca, y_tr_fold)
            fit_sec = perf_counter() - t0
            pred = knn.predict(X_va_pca)

        acc = float(accuracy_score(y_va_fold, pred))
        f1 = float(f1_score(y_va_fold, pred, average='macro', zero_division=0))

        cv_records.append({
            'experiment': exp_name,
            'fold': fold_idx,
            'accuracy': acc,
            'macro_f1': f1,
            'fit_seconds': fit_sec,
            'pca_fit_seconds': pca_sec
        })

    return cv_records


def select_best(candidates):
    """Chọn mọi ứng viên theo validation F1, rồi accuracy; giữ thứ tự khi hòa."""
    if not candidates:
        raise ValueError('Không có ứng viên để chọn')
    return max(candidates, key=lambda row: (row['val_macro_f1'], row['val_accuracy']))


def build_unified_comparison():
    """Build unified comparison table across E1–E7 and ablation variants."""
    v2_comp = json.loads((ROOT / 'results/v2/metrics/comparison.json').read_text())
    v3_cv = json.loads((ROOT / 'results/v3/metrics/cv_summary.json').read_text())
    v5_val = json.loads((OUT / 'validation_search.json').read_text())
    v5_winners = json.loads((OUT / 'winners.json').read_text())
    v5_cv = json.loads((OUT / 'cv.json').read_text())

    # Map CV summaries from V3
    cv_map = {}
    for s in v3_cv['summaries']:
        exp = s['experiment']
        if exp == 'E6' and s.get('pca_seed') != 42:
            continue
        cv_map[exp] = {
            'cv_macro_f1_mean': s['mean_macro_f1'],
            'cv_macro_f1_std': s.get('std_macro_f1_population', 0.0)
        }

    # Add V5 CV
    v5_cv_df = pd.DataFrame(v5_cv)
    for exp, grp in v5_cv_df.groupby('experiment'):
        cv_map[exp] = {
            'cv_macro_f1_mean': float(grp['macro_f1'].mean()),
            'cv_macro_f1_std': float(grp['macro_f1'].std(ddof=0)),
            'cv_accuracy_mean': float(grp['accuracy'].mean()),
            'cv_accuracy_std': float(grp['accuracy'].std(ddof=0))
        }

    unified = []
    # E1–E6
    for e in v2_comp['experiments']:
        exp_id = e['experiment']
        unified.append({
            'experiment': exp_id,
            'variant': e['variant'],
            'params': e['params'],
            'pca_components': e.get('pca_components'),
            'val_accuracy': e['val_accuracy'],
            'val_macro_f1': e['val_macro_f1'],
            'cv_macro_f1_mean': cv_map.get(exp_id, {}).get('cv_macro_f1_mean'),
            'cv_macro_f1_std': cv_map.get(exp_id, {}).get('cv_macro_f1_std'),
            'fit_seconds': e.get('fit_seconds', 0.0) + e.get('pca_fit_seconds', 0.0)
        })

    # E7 winner
    e7_winner = next(w for w in v5_winners if w['experiment'] == 'E7')
    unified.append({
        'experiment': 'E7',
        'variant': e7_winner['feature'],
        'params': e7_winner['params'],
        'pca_components': None,
        'val_accuracy': e7_winner['accuracy'],
        'val_macro_f1': e7_winner['macro_f1'],
        'cv_macro_f1_mean': cv_map.get('E7', {}).get('cv_macro_f1_mean'),
        'cv_macro_f1_std': cv_map.get('E7', {}).get('cv_macro_f1_std'),
        'fit_seconds': e7_winner['fit_seconds']
    })

    # Controlled variants
    for w in v5_winners:
        if w['experiment'] in ('E6_size128', 'E6_bins32'):
            exp_id = w['experiment']
            unified.append({
                'experiment': exp_id,
                'variant': w['feature'],
                'params': w['params'],
                'pca_components': 128,
                'val_accuracy': w['accuracy'],
                'val_macro_f1': w['macro_f1'],
                'cv_macro_f1_mean': cv_map.get(exp_id, {}).get('cv_macro_f1_mean'),
                'cv_macro_f1_std': cv_map.get(exp_id, {}).get('cv_macro_f1_std'),
                'fit_seconds': w['fit_seconds'] + w['pca_fit_seconds']
            })

    # Tất cả cấu hình trong bảng, kể cả đối chứng, đều thuộc tập ứng viên.
    selected = select_best(unified)

    out_data = {
        'unified_comparison': unified,
        'selected_model': selected['experiment'],
        'selection_rule': 'validation macro F1, then accuracy',
        'pretrained_extension_status': 'skipped: No torch/torchvision in locked environment'
    }
    save_json(OUT / 'unified_comparison.json', out_data)
    return out_data


def run():
    if (OUT / 'validation_search.json').exists():
        raise FileExistsError('Đã có kết quả V5; không tự ghi đè. Dùng bản sao cách ly để tái lập.')
    cfg = load_config()
    df = load_dataset()
    selected, arrays = extract_or_load_features(df, cfg)

    train_mask = (arrays['splits'] == 'train')
    val_mask = (arrays['splits'] == 'val')
    y_train = arrays['labels'][train_mask]
    y_val = arrays['labels'][val_mask]

    # E7 search
    e7_results = run_e7_grid(arrays['h8_hsv'][train_mask], y_train,
                             arrays['h8_hsv'][val_mask], y_val, cfg)
    
    # Controlled experiments on validation
    ctrl_results = run_controlled_val(arrays, y_train, y_val, arrays['splits'], cfg)
    
    all_val = e7_results + ctrl_results
    save_json(OUT / 'validation_search.json', all_val)

    # Winners
    best_e7 = sorted(e7_results, key=lambda x: (x['macro_f1'], x['accuracy']), reverse=True)[0]
    winners = [best_e7] + ctrl_results
    save_json(OUT / 'winners.json', winners)

    # 5-fold CV on train only
    cv_records = []
    cv_records.extend(run_5fold_cv(arrays['h8_hsv'][train_mask], y_train, 'E7', best_e7, cfg))
    cv_records.extend(run_5fold_cv(arrays['E6_size128'][train_mask], y_train, 'E6_size128',
                                  cfg['controlled_experiments']['E6_size128'], cfg))
    cv_records.extend(run_5fold_cv(arrays['E6_bins32'][train_mask], y_train, 'E6_bins32',
                                  cfg['controlled_experiments']['E6_bins32'], cfg))
    save_json(OUT / 'cv.json', cv_records)

    unified = build_unified_comparison()
    print("Pipeline V5 finished successfully. Selected model:", unified['selected_model'])


if __name__ == '__main__':
    run()
