"""Reproducible, staged baseline workflow (run with python -m src.pipeline)."""
import argparse
import hashlib
import json
import pickle
import importlib.metadata
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from .data_loader import load_dataset
from .preprocessing import preprocess
from .features import extract
from .models import search
from .evaluation import metrics, plot_confusion
from .utils import ROOT, config, digest, save_json

RAW = ROOT / 'data/raw'
PROC = ROOT / 'data/processed'
FIG = ROOT / 'results/figures'
MET = ROOT / 'results/metrics'
MOD = ROOT / 'results/models'

def prepare_dirs():
    for p in (PROC, FIG, MET, MOD): p.mkdir(parents=True, exist_ok=True)

def data():
    """Validate all images and export relative metadata for auditability."""
    prepare_dirs(); df = load_dataset()
    exported = df.copy(); exported['path'] = exported.path.map(lambda p: str(Path(p).relative_to(ROOT)))
    exported.to_csv(MET / 'dataset_metadata.csv', index=False)
    print('Counts:', df.groupby('split').size().to_dict(), 'classes:', df.label.nunique())
    return df

def eda(df):
    """Save dataset distributions and representative images."""
    print(df.groupby('split').size()); print('Image modes:', df['mode'].value_counts().to_dict()); print(df[['width','height']].describe())
    fig, ax = plt.subplots(figsize=(16, 4)); df.groupby('class_id').size().plot.bar(ax=ax, width=1); ax.set(xlabel='Official class ID', ylabel='Images', title='Class distribution'); ax.tick_params(axis='x', labelsize=5); fig.tight_layout(); fig.savefig(FIG/'class_distribution.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ax[0].scatter(df.width, df.height, s=2, alpha=.2); ax[0].set(xlabel='Width', ylabel='Height'); (df.width / df.height).hist(ax=ax[1], bins=40); ax[1].set(xlabel='Aspect ratio', ylabel='Count'); fig.tight_layout(); fig.savefig(FIG/'dimensions.png', dpi=150); plt.close(fig)
    sample = df.groupby('class_id').head(1).head(12); fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, row in zip(axes.flat, sample.itertuples()):
        with Image.open(row.path) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f'Class {row.class_id}'); ax.axis('off')
    fig.tight_layout(); fig.savefig(FIG/'samples.png', dpi=150); plt.close(fig)

def preview(df, cfg):
    """Save an original and its deterministic preprocessing result."""
    row = df.iloc[0]; fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    with Image.open(row.path) as im: ax[0].imshow(im.convert('RGB'))
    ax[1].imshow(preprocess(Path(row.path), cfg['image_size'])); ax[0].set_title('Original'); ax[1].set_title('RGB, resized and normalized')
    for a in ax: a.axis('off')
    fig.tight_layout(); fig.savefig(FIG/'preprocessing.png', dpi=150); plt.close(fig)

def feature_cache(df, cfg):
    """Cache features keyed by config, metadata and image content, with IDs for alignment."""
    provenance = {'image_size': cfg['image_size'], 'hog': cfg['hog'], 'code': {name: digest(ROOT / 'src' / name) for name in ('preprocessing.py', 'features.py')}, 'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-image', 'Pillow')}}
    h = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode())
    for row in df.itertuples():
        h.update(f'{row.image_id}:{row.label}:{row.split}:'.encode()); h.update(digest(Path(row.path)).encode())
    signature = h.hexdigest(); cache = PROC / 'features.npz'; manifest = PROC / 'features.json'
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()).get('signature') == signature:
        print('Using verified feature cache'); return
    pixels, hogs = [], []
    for row in df.itertuples():
        image = preprocess(Path(row.path), cfg['image_size']); pixels.append(extract(image, 'pixels', cfg)); hogs.append(extract(image, 'hog', cfg))
    np.savez_compressed(cache, ids=df.image_id.to_numpy(), labels=df.label.to_numpy(), splits=df.split.to_numpy(dtype=str), pixels=np.stack(pixels), hog=np.stack(hogs))
    save_json(manifest, {'signature': signature, 'cache_sha256': digest(cache), 'settings': {'image_size': cfg['image_size'], 'hog': cfg['hog']}})
    print('Feature shapes:', np.shape(pixels), np.shape(hogs))

def load_features(df):
    """Reject stale, corrupt or misaligned feature artifacts."""
    cache = PROC/'features.npz'; manifest = json.loads((PROC/'features.json').read_text())
    if digest(cache) != manifest['cache_sha256']: raise ValueError('Feature cache checksum mismatch; delete cache and regenerate.')
    with np.load(cache, allow_pickle=False) as f: out = {k: f[k] for k in f.files}
    for key, expected in [('ids', df.image_id.to_numpy()), ('labels', df.label.to_numpy()), ('splits', df.split.to_numpy())]:
        if not np.array_equal(out[key], expected): raise ValueError(f'Feature cache {key} does not align with metadata.')
    for key in ('pixels', 'hog'):
        if out[key].ndim != 2 or len(out[key]) != len(df) or not np.isfinite(out[key]).all(): raise ValueError(f'Invalid {key} features')
    return out

def train(df, cfg):
    """Search all configurations on validation and persist the single selected model."""
    f = load_features(df); train_mask = f['splits'] == 'train'; val_mask = f['splits'] == 'val'
    results, winners = [], []
    for exp_id, spec in cfg['experiments'].items():
        X = f[spec['features']]; model, records = search(X[train_mask], f['labels'][train_mask], X[val_mask], f['labels'][val_mask], spec, cfg['seed'])
        for record in records: results.append({'experiment': exp_id, 'features': spec['features'], 'model': spec['model'], **record})
        pred = model.predict(X[val_mask]); score = metrics(f['labels'][val_mask], pred)
        winners.append((score['macro_f1'], score['accuracy'], exp_id, model, score))
    # Sort descending on scores; stable configuration order resolves exact ties.
    best = sorted(winners, key=lambda item: (item[0], item[1]), reverse=True)[0]
    _, _, exp_id, model, val_metrics = best
    payload = {'experiment': exp_id, 'model': model, 'config': cfg, 'feature_signature': json.loads((PROC/'features.json').read_text())['signature']}
    with (MOD/'selected_model.pkl').open('wb') as out: pickle.dump(payload, out)
    save_json(MET/'validation_search.json', results)
    save_json(MET/'selection.json', {'experiment': exp_id, 'params': model.get_params(), 'validation_metrics': val_metrics, 'selection_rule': 'highest validation macro F1; accuracy tie-break; config order final tie-break', 'model_sha256': digest(MOD/'selected_model.pkl'), 'config_sha256': digest(ROOT/'configs/experiments.yaml'), 'feature_signature': payload['feature_signature']})
    print('Selected:', exp_id, val_metrics)

def selected(cfg):
    """Load only locally produced model and verify its provenance."""
    sel = json.loads((MET/'selection.json').read_text()); path = MOD/'selected_model.pkl'
    if digest(path) != sel['model_sha256'] or digest(ROOT/'configs/experiments.yaml') != sel['config_sha256'] or json.loads((PROC/'features.json').read_text())['signature'] != sel['feature_signature']:
        raise ValueError('Model, configuration or feature provenance changed; rerun training.')
    # Never load an untrusted pickle.
    with path.open('rb') as f: payload = pickle.load(f)
    if payload['experiment'] != sel['experiment'] or payload['config'] != cfg or payload['feature_signature'] != sel['feature_signature'] or payload['model'].get_params() != sel['params']:
        raise ValueError('Saved model does not match selection record.')
    return payload

def evaluate(df, cfg):
    """Evaluate the validation-selected train-only model once on the official test split."""
    f = load_features(df); p = selected(cfg); mask = f['splits'] == 'test'; X = f[cfg['experiments'][p['experiment']]['features']][mask]; y = f['labels'][mask]
    start = perf_counter(); pred = p['model'].predict(X); elapsed = perf_counter() - start
    if not ((pred >= 0) & (pred < 102)).all(): raise ValueError('Prediction outside class range')
    result = {'experiment': p['experiment'], 'split': 'test', 'metrics': metrics(y, pred), 'predict_seconds': elapsed, 'model_sha256': digest(MOD/'selected_model.pkl')}
    save_json(MET/'test_metrics.json', result)
    pd.DataFrame({'image_id': f['ids'][mask], 'true_class_id': y+1, 'predicted_class_id': pred+1}).to_csv(MET/'test_predictions.csv', index=False)
    cm = plot_confusion(y, pred, FIG/'confusion_matrix.png'); np.save(MET/'confusion_matrix.npy', cm)
    print(result)

def errors(df):
    """Save frequent confusions and a deterministic sample of incorrect predictions."""
    pred = pd.read_csv(MET/'test_predictions.csv'); wrong = pred[pred.true_class_id != pred.predicted_class_id]
    pairs = wrong.groupby(['true_class_id', 'predicted_class_id']).size().sort_values(ascending=False).rename('count'); pairs.to_csv(MET/'confused_pairs.csv')
    print('Most frequent confusions:', pairs.head(10)); print('Errors:', len(wrong))
    sample = wrong.head(12).merge(df[['image_id', 'path']], on='image_id', validate='one_to_one')
    if sample.empty:
        (FIG/'errors.png').unlink(missing_ok=True)
        print('No errors to display.'); return
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat: ax.axis('off')
    for ax, row in zip(axes.flat, sample.itertuples()):
        with Image.open(row.path) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f'True {row.true_class_id}, predicted {row.predicted_class_id}')
    fig.tight_layout(); fig.savefig(FIG/'errors.png', dpi=150); plt.close(fig)

def sanity(df, cfg):
    """Check dataset, feature alignment, artifact provenance and prediction range."""
    f = load_features(df); p = selected(cfg)
    assert set(f['labels']) == set(range(102))
    assert len(f['ids']) == len(set(f['ids'])) == 8189
    assert set(f['splits']) == {'train', 'val', 'test'}
    assert all(f[k].shape[1] > 0 for k in ('pixels', 'hog'))
    assert p['config']['seed'] == cfg['seed']
    prediction = pd.read_csv(MET/'test_predictions.csv'); expected = df[df.split == 'test']
    assert np.array_equal(prediction.image_id.to_numpy(), expected.image_id.to_numpy())
    assert np.array_equal(prediction.true_class_id.to_numpy(), expected.class_id.to_numpy())
    assert prediction.predicted_class_id.between(1, 102).all()
    saved_metrics = json.loads((MET/'test_metrics.json').read_text())
    assert saved_metrics['model_sha256'] == digest(MOD/'selected_model.pkl')
    assert saved_metrics['metrics'] == metrics(prediction.true_class_id-1, prediction.predicted_class_id-1)
    print('Sanity checks passed; training-only fitting enforced by train() and search().')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['data', 'eda', 'preprocess', 'features', 'train', 'evaluate', 'errors', 'sanity', 'all']); args = parser.parse_args()
    cfg = config(); df = data()
    if args.stage in ('eda', 'all'): eda(df)
    if args.stage in ('preprocess', 'all'): preview(df, cfg)
    if args.stage in ('features', 'all'): feature_cache(df, cfg)
    if args.stage in ('train', 'evaluate', 'sanity', 'all'): feature_cache(df, cfg)
    if args.stage in ('train', 'all'): train(df, cfg)
    if args.stage in ('evaluate', 'all'): evaluate(df, cfg)
    if args.stage in ('errors', 'all'): errors(df)
    if args.stage in ('sanity', 'all'): sanity(df, cfg)
if __name__ == '__main__': main()
