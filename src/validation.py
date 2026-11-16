"""Đánh giá và phân tích validation V1; không dự đoán test."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from . import pipeline as p
from .evaluation import metrics, plot_confusion
from .utils import save_json


def evaluate_validation(df: pd.DataFrame, cfg: dict) -> dict:
    """Dùng model đã chọn để lưu chỉ số và dự đoán validation."""
    features = p.load_features(df); payload = p.selected(cfg)
    mask = features['splits'] == 'val'
    if not mask.any(): raise ValueError('Không có mẫu validation trong feature')
    X = features[cfg['experiments'][payload['experiment']]['features']][mask]
    labels = features['labels'][mask]; predictions = payload['model'].predict(X)
    if predictions.shape != labels.shape or not ((predictions >= 0) & (predictions < 102)).all():
        raise ValueError('Dự đoán validation không hợp lệ')
    result = {'experiment': payload['experiment'], 'split': 'val', 'metrics': metrics(labels, predictions)}
    save_json(p.MET/'validation_metrics.json', result)
    pd.DataFrame({'image_id': features['ids'][mask], 'true_class_id': labels+1, 'predicted_class_id': predictions+1}).to_csv(p.MET/'validation_predictions.csv', index=False)
    cm = plot_confusion(labels, predictions, p.FIG/'validation_confusion_matrix.png', title='Ma trận nhầm lẫn validation')
    np.save(p.MET/'validation_confusion_matrix.npy', cm)
    return result


def validation_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Lưu các cặp nhầm và ví dụ lỗi từ dự đoán validation đã tạo."""
    pred = pd.read_csv(p.MET/'validation_predictions.csv')
    expected = df[df.split == 'val']
    if not np.array_equal(pred.image_id.to_numpy(), expected.image_id.to_numpy()) or not np.array_equal(pred.true_class_id.to_numpy(), expected.class_id.to_numpy()) or not pred.predicted_class_id.between(1, 102).all():
        raise ValueError('Dự đoán không khớp các ảnh validation chính thức')
    errors = pred[pred.true_class_id != pred.predicted_class_id]
    pairs = errors.groupby(['true_class_id','predicted_class_id']).size().sort_values(ascending=False).rename('count')
    pairs.to_csv(p.MET/'validation_confused_pairs.csv')
    sample = errors.head(12).merge(df[['image_id','path']], on='image_id', validate='one_to_one')
    if not sample.empty:
        fig, axes = plt.subplots(3, 4, figsize=(12,9))
        for ax in axes.flat: ax.axis('off')
        for ax, row in zip(axes.flat, sample.itertuples()):
            with Image.open(Path(row.path)) as image: ax.imshow(image.convert('RGB'))
            ax.set_title(f'Thật {row.true_class_id}, dự đoán {row.predicted_class_id}')
        fig.tight_layout(); fig.savefig(p.FIG/'validation_errors.png', dpi=150); plt.close(fig)
    else:
        (p.FIG/'validation_errors.png').unlink(missing_ok=True)
    return pairs.head(10).reset_index()


def sanity_validation(df: pd.DataFrame, cfg: dict) -> None:
    """Kiểm tra dataset, feature, model và dự đoán validation; không cần test."""
    f = p.load_features(df); payload = p.selected(cfg)
    if len(f['ids']) != 8189 or len(set(f['ids'])) != 8189 or set(f['labels']) != set(range(102)):
        raise ValueError('ID hoặc nhãn không hợp lệ')
    pred = pd.read_csv(p.MET/'validation_predictions.csv')
    expected = df[df.split == 'val']
    if not np.array_equal(pred.image_id.to_numpy(), expected.image_id.to_numpy()) or not np.array_equal(pred.true_class_id.to_numpy(), expected.class_id.to_numpy()):
        raise ValueError('Dự đoán validation không khớp dữ liệu')
    if not pred.predicted_class_id.between(1,102).all(): raise ValueError('Mã dự đoán ngoài khoảng 1–102')
    result = json.loads((p.MET/'validation_metrics.json').read_text())
    if result['experiment'] != payload['experiment'] or result['metrics'] != metrics(pred.true_class_id-1, pred.predicted_class_id-1):
        raise ValueError('Chỉ số validation không khớp dự đoán')
    print('Các kiểm tra validation đã pass; không đánh giá test.')
