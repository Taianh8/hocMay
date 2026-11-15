"""Đánh giá test cuối duy nhất cho bài toán Oxford 102 Flowers.

Thực hiện:
1. Huấn luyện lại Baseline E1 và Mô hình cuối E6 trên tập train + validation (2040 ảnh).
2. Phép biến đổi PCA chỉ fit trên train + validation (2040 ảnh), sau đó transform test.
3. Dự đoán test ĐÚNG MỘT LẦN (6149 ảnh).
4. Tính chỉ số toàn bộ test và phân tích độ nhạy (loại ảnh trùng 7309, 8067).
5. Lưu kết quả vào results/final_test/ và tạo configs/e6_manifest_v2.json.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import load_dataset
from src.features_v2 import hog_color
from src.flower_catalog import load_catalog
from src.preprocessing import preprocess
from src.utils import ROOT, digest, save_json

OUT = ROOT / 'results/final_test'
OUT_MODELS = OUT / 'models'


def extract_features(df):
    """Trích xuất đặc trưng pixel (E1) và HOG+HSV (E6) độc lập cho từng ảnh."""
    h8_settings = {'orientations': 9, 'pixels_per_cell': 8, 'cells_per_block': 2}
    pixels_list = []
    h8_hsv_list = []
    
    print(f"Trích xuất đặc trưng cho {len(df)} ảnh...", flush=True)
    start = perf_counter()
    for row in df.itertuples():
        p = Path(row.path)
        im = preprocess(p, 64)
        pixels_list.append(im.ravel().astype(np.float32))
        h8_hsv_list.append(hog_color(im, h8_settings, 'hsv', 16))
    
    elapsed = perf_counter() - start
    print(f"Hoàn thành trích xuất trong {elapsed:.2f}s", flush=True)
    return np.stack(pixels_list), np.stack(h8_hsv_list)


def compute_metrics_dict(y_true, y_pred):
    """Tính các chỉ số tổng hợp."""
    acc = float(accuracy_score(y_true, y_pred))
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'accuracy': acc,
        'macro_precision': float(p_m),
        'macro_recall': float(r_m),
        'macro_f1': float(f1_m),
        'weighted_precision': float(p_w),
        'weighted_recall': float(r_w),
        'weighted_f1': float(f1_w),
        'samples': int(len(y_true))
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print("Tổng số ảnh:", len(df))
    print("Phân bố split:", df['split'].value_counts().to_dict())

    # Tách train+val và test
    trainval_mask = df['split'].isin(['train', 'val']).to_numpy()
    test_mask = (df['split'] == 'test').to_numpy()

    df_trainval = df[trainval_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    y_trainval = df_trainval['label'].to_numpy()
    y_test = df_test['label'].to_numpy()
    test_ids = df_test['image_id'].to_numpy()

    # Trích xuất đặc trưng
    cache_file = OUT / 'features_all.npz'
    if cache_file.exists():
        print("Đọc cache đặc trưng từ", cache_file)
        with np.load(cache_file, allow_pickle=False) as f:
            pixels = f['pixels']
            h8_hsv = f['h8_hsv']
    else:
        pixels, h8_hsv = extract_features(df)
        np.savez_compressed(cache_file, pixels=pixels, h8_hsv=h8_hsv)

    X_trainval_pix = pixels[trainval_mask]
    X_test_pix = pixels[test_mask]

    X_trainval_h8 = h8_hsv[trainval_mask]
    X_test_h8 = h8_hsv[test_mask]

    # --- 1. BASELINE E1: Pixels + 1-NN (Euclidean) ---
    print("\n--- Huấn luyện và đánh giá E1 (Baseline) trên test ---")
    knn_e1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean', weights='uniform')
    t0 = perf_counter()
    knn_e1.fit(X_trainval_pix, y_trainval)
    e1_fit_time = perf_counter() - t0

    t0 = perf_counter()
    pred_e1 = knn_e1.predict(X_test_pix)
    e1_pred_time = perf_counter() - t0

    e1_metrics_full = compute_metrics_dict(y_test, pred_e1)
    e1_metrics_full['fit_seconds'] = e1_fit_time
    e1_metrics_full['predict_seconds'] = e1_pred_time

    # Lưu dự đoán E1
    e1_pred_df = pd.DataFrame({
        'image_id': test_ids,
        'true_class_id': y_test + 1,
        'pred_class_id': pred_e1 + 1,
        'correct': (y_test == pred_e1)
    })
    e1_pred_df.to_csv(OUT / 'E1_test_predictions.csv', index=False)

    cm_e1 = confusion_matrix(y_test, pred_e1, labels=range(102))
    np.save(OUT / 'E1_confusion_matrix.npy', cm_e1)

    # --- 2. FINAL MODEL E6: HOG+HSV + PCA(128) + 3-NN (Cosine, Distance) ---
    print("\n--- Huấn luyện và đánh giá E6 (Mô hình cuối) trên test ---")
    pca = PCA(n_components=128, svd_solver='randomized', iterated_power=3, random_state=42)
    t0 = perf_counter()
    pca.fit(X_trainval_h8)  # Fit PCA CHỈ trên train+val!
    pca_fit_time = perf_counter() - t0

    X_trainval_pca = pca.transform(X_trainval_h8)
    t0 = perf_counter()
    X_test_pca = pca.transform(X_test_h8)
    pca_transform_time = perf_counter() - t0

    knn_e6 = KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance')
    t0 = perf_counter()
    knn_e6.fit(X_trainval_pca, y_trainval)
    e6_fit_time = perf_counter() - t0

    t0 = perf_counter()
    pred_e6 = knn_e6.predict(X_test_pca)
    e6_pred_time = perf_counter() - t0

    e6_metrics_full = compute_metrics_dict(y_test, pred_e6)
    e6_metrics_full['pca_fit_seconds'] = pca_fit_time
    e6_metrics_full['pca_test_transform_seconds'] = pca_transform_time
    e6_metrics_full['knn_fit_seconds'] = e6_fit_time
    e6_metrics_full['knn_predict_seconds'] = e6_pred_time
    e6_metrics_full['total_fit_seconds'] = pca_fit_time + e6_fit_time

    # Lưu dự đoán E6
    e6_pred_df = pd.DataFrame({
        'image_id': test_ids,
        'true_class_id': y_test + 1,
        'pred_class_id': pred_e6 + 1,
        'correct': (y_test == pred_e6)
    })
    e6_pred_df.to_csv(OUT / 'E6_test_predictions.csv', index=False)

    cm_e6 = confusion_matrix(y_test, pred_e6, labels=range(102))
    np.save(OUT / 'E6_confusion_matrix.npy', cm_e6)

    # --- 3. ĐỘ NHẠY: LOẠI ẢNH TRÙNG BYTE (7309 và 8067) ---
    dup_ids = {7309, 8067}
    clean_mask = ~np.isin(test_ids, list(dup_ids))
    print(f"\nPhân tích độ nhạy: loại {len(dup_ids)} ảnh trùng byte, còn {clean_mask.sum()} ảnh test.")

    y_test_clean = y_test[clean_mask]
    pred_e1_clean = pred_e1[clean_mask]
    pred_e6_clean = pred_e6[clean_mask]

    e1_metrics_clean = compute_metrics_dict(y_test_clean, pred_e1_clean)
    e6_metrics_clean = compute_metrics_dict(y_test_clean, pred_e6_clean)

    # Chi tiết về 2 ảnh trùng
    dup_details = []
    for dup_id in sorted(dup_ids):
        idx = np.where(test_ids == dup_id)[0][0]
        dup_details.append({
            'test_image_id': int(dup_id),
            'true_class_id': int(y_test[idx] + 1),
            'e1_pred_class_id': int(pred_e1[idx] + 1),
            'e1_correct': bool(pred_e1[idx] == y_test[idx]),
            'e6_pred_class_id': int(pred_e6[idx] + 1),
            'e6_correct': bool(pred_e6[idx] == y_test[idx])
        })

    # --- 4. PER-CLASS VÀ CONFUSION PAIRS CHO E6 ---
    catalog = load_catalog()
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_test, pred_e6, labels=range(102), zero_division=0
    )
    per_class_records = []
    for c in range(102):
        cid = c + 1
        name = catalog[cid]['name_en']
        per_class_records.append({
            'class_id': cid,
            'name_en': name,
            'name_vi': catalog[cid].get('name_vi'),
            'support': int(support_class[c]),
            'precision': float(p_class[c]),
            'recall': float(r_class[c]),
            'f1': float(f1_class[c])
        })
    per_class_df = pd.DataFrame(per_class_records)
    per_class_df.to_csv(OUT / 'E6_per_class.csv', index=False)

    # Confusion pairs (true != pred)
    confusion_pairs = []
    for i in range(102):
        for j in range(102):
            if i != j and cm_e6[i, j] > 0:
                confusion_pairs.append({
                    'true_class_id': i + 1,
                    'true_name_en': catalog[i + 1]['name_en'],
                    'pred_class_id': j + 1,
                    'pred_name_en': catalog[j + 1]['name_en'],
                    'count': int(cm_e6[i, j])
                })
    pairs_df = pd.DataFrame(confusion_pairs).sort_values(by='count', ascending=False)
    pairs_df.to_csv(OUT / 'E6_confusion_pairs.csv', index=False)

    # Lưu toàn bộ chỉ số vào metrics.json
    metrics_summary = {
        'protocol': 'Chỉ chạy đúng một lần trên test sau khi đã chốt mô hình cuối trên validation/CV',
        'training_split': 'train + val (2040 images)',
        'test_split': 'test (6149 images)',
        'duplicate_images_excluded_in_sensitivity': list(dup_ids),
        'duplicate_details': dup_details,
        'E1_baseline': {
            'full_test': e1_metrics_full,
            'clean_test_sensitivity': e1_metrics_clean
        },
        'E6_final_model': {
            'full_test': e6_metrics_full,
            'clean_test_sensitivity': e6_metrics_clean
        }
    }
    save_json(OUT / 'metrics.json', metrics_summary)

    # Lưu model đã huấn luyện lại
    with (OUT_MODELS / 'E6_retrained.pkl').open('wb') as f:
        pickle.dump({
            'experiment': 'E6',
            'pca': pca,
            'model': knn_e6,
            'training_samples': 2040,
            'classes': np.arange(102)
        }, f)
    with (OUT_MODELS / 'E1_retrained.pkl').open('wb') as f:
        pickle.dump({
            'experiment': 'E1',
            'model': knn_e1,
            'training_samples': 2040,
            'classes': np.arange(102)
        }, f)

    # --- 5. TẠO configs/e6_manifest_v2.json ---
    create_v2_manifest(metrics_summary)
    print("\nHoàn thành đánh giá test cuối! Kết quả lưu tại results/final_test/")


def create_v2_manifest(metrics_summary):
    manifest_path = ROOT / 'configs/e6_manifest_v2.json'
    tracked_files = [
        'configs/experiments.yaml',
        'configs/experiments_v2.yaml',
        'configs/experiments_v3.yaml',
        'configs/experiments_v5.yaml',
        'src/preprocessing.py',
        'src/features_v2.py',
        'src/data_loader.py',
        'src/flower_catalog.py',
        'results/v5/unified_comparison.json',
        'results/final_test/metrics.json',
        'results/final_test/E6_test_predictions.csv',
        'results/final_test/E6_per_class.csv',
        'results/final_test/E6_confusion_pairs.csv',
        'data/raw/imagelabels.mat',
        'data/raw/setid.mat',
        'requirements-v4-lock.txt'
    ]
    file_hashes = {p: digest(ROOT / p) for p in tracked_files if (ROOT / p).is_file()}
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'scikit-image', 'Pillow')}

    manifest = {
        'manifest_version': 2,
        'experiment': 'E6',
        'baseline': 'E1',
        'selection': {
            'rule': 'validation macro F1, then accuracy',
            'validation_macro_f1': 0.21266311022542148,
            'validation_accuracy': 0.23431372549019608,
            'cv_macro_f1_mean': 0.19316738816738818,
            'cv_macro_f1_std': 0.019501900887704513
        },
        'final_test': {
            'evaluation_status': 'completed',
            'training_set': 'train + validation (2040 samples)',
            'test_set': 'official test (6149 samples)',
            'E6_metrics': metrics_summary['E6_final_model']['full_test'],
            'E6_clean_sensitivity': metrics_summary['E6_final_model']['clean_test_sensitivity'],
            'E1_metrics': metrics_summary['E1_baseline']['full_test'],
            'E1_clean_sensitivity': metrics_summary['E1_baseline']['clean_test_sensitivity']
        },
        'dataset': {
            'name': 'Oxford 102 Flowers',
            'images': 8189,
            'classes': 102,
            'splits': {'train': 1020, 'validation': 1020, 'test': 6149},
            'mapping': 'official class ID = internal predicted index + 1'
        },
        'preprocessing': {
            'mode': 'RGB',
            'size': [64, 64],
            'resampling': 'Pillow bilinear',
            'normalization': 'float32 / 255',
            'exif_transpose': False
        },
        'features': {
            'variant': 'h8_hsv',
            'orientations': 9,
            'pixels_per_cell': 8,
            'cells_per_block': 2,
            'block_norm': 'L2-Hys',
            'histogram_bins_per_channel': 16,
            'block_normalization': 'each block L2 normalized; concatenated equal weight',
            'input_dimensions': 1812
        },
        'pca': {
            'components': 128,
            'svd_solver': 'randomized',
            'iterated_power': 3,
            'random_seed': 42,
            'training_samples': 2040
        },
        'knn': {
            'n_neighbors': 3,
            'metric': 'cosine',
            'weights': 'distance',
            'n_jobs': 1,
            'training_samples': 2040
        },
        'python': platform.python_version(),
        'dependencies': versions,
        'complete_dependency_lock': 'requirements-v4-lock.txt',
        'sha256': file_hashes,
        'model_artifact': 'results/final_test/models/E6_retrained.pkl',
        'trust': 'Local trusted pickle only; hashes verify reproducibility and provenance.'
    }
    save_json(manifest_path, manifest)
    print(f"Đã tạo manifest: {manifest_path}")


if __name__ == '__main__':
    main()
