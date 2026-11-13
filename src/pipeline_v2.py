"""Validation-only classical feature experiments, isolated from V1 artifacts."""
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from .data_loader import load_dataset
from .evaluation import metrics
from .features_v2 import color_histogram, hog_feature
from .models import search
from .pipeline import load_features
from .preprocessing import preprocess
from .utils import ROOT, config as v1_config, digest, save_json

OUT = ROOT / 'results/v2'
V1_MET = ROOT / 'results/metrics'
V1_PROC = ROOT / 'data/processed'
V2_CACHE = V1_PROC / 'features_v2.npz'
V2_MANIFEST = V1_PROC / 'features_v2.json'


def load_config():
    """Read V2 settings independently so V1 provenance remains unchanged."""
    with (ROOT / 'configs/experiments_v2.yaml').open() as f:
        return yaml.safe_load(f)


def checked_v1_features(df, cfg):
    """Check source image and code provenance before reading V1 cached features."""
    provenance = {'image_size': cfg['image_size'], 'hog': cfg['hog'],
                  'code': {name: digest(ROOT / 'src' / name) for name in ('preprocessing.py', 'features.py')},
                  'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-image', 'Pillow')}}
    h = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode())
    for row in df.itertuples():
        h.update(f'{row.image_id}:{row.label}:{row.split}:'.encode())
        h.update(digest(Path(row.path)).encode())
    manifest = json.loads((V1_PROC / 'features.json').read_text())
    if h.hexdigest() != manifest['signature']:
        raise ValueError('V1 input or code provenance differs from the stored cache; reproduce V1 separately before V2.')
    return load_features(df)


def v2_features(df, cfg):
    """Extract and cache features only for official train and validation images."""
    selected = df[df.split.isin(['train', 'val'])].reset_index(drop=True)
    provenance = {'settings': {k: cfg[k] for k in ('image_size', 'hog_variants', 'feature_variants', 'histogram_bins')},
                  'code': {name: digest(ROOT / 'src' / name) for name in ('preprocessing.py', 'features_v2.py')},
                  'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-image', 'Pillow')}}
    h = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode())
    for row in selected.itertuples():
        h.update(f'{row.image_id}:{row.label}:{row.split}:'.encode()); h.update(digest(Path(row.path)).encode())
    signature = h.hexdigest()
    if V2_CACHE.exists() and V2_MANIFEST.exists():
        manifest = json.loads(V2_MANIFEST.read_text())
        if manifest['signature'] == signature and manifest['cache_sha256'] == digest(V2_CACHE):
            with np.load(V2_CACHE, allow_pickle=False) as f:
                arrays = {k: f[k] for k in f.files}
            for key, expected in [('ids', selected.image_id.to_numpy()), ('labels', selected.label.to_numpy()), ('splits', selected.split.to_numpy())]:
                if not np.array_equal(arrays[key], expected): raise ValueError(f'V2 cache {key} alignment mismatch')
            print('Using V2 feature cache', flush=True)
            return selected, arrays, manifest['extraction_seconds']
    start = perf_counter()
    extracted = {v['id']: [] for v in cfg['feature_variants']}
    for row in selected.itertuples():
        image = preprocess(Path(row.path), cfg['image_size'])
        edges = {key: hog_feature(image, settings) for key, settings in cfg['hog_variants'].items()}
        colors = {space: color_histogram(image, space, cfg['histogram_bins']) for space in {v['color'] for v in cfg['feature_variants']}}
        for variant in cfg['feature_variants']:
            extracted[variant['id']].append(np.concatenate((edges[variant['hog']], colors[variant['color']])))
    arrays = {key: np.stack(value) for key, value in extracted.items()}
    arrays.update(ids=selected.image_id.to_numpy(), labels=selected.label.to_numpy(), splits=selected.split.to_numpy(dtype=str))
    elapsed = perf_counter() - start
    np.savez_compressed(V2_CACHE, **arrays)
    save_json(V2_MANIFEST, {'signature': signature, 'cache_sha256': digest(V2_CACHE), 'extraction_seconds': elapsed, 'provenance': provenance})
    print(f'Extracted V2 features in {elapsed:.2f}s', flush=True)
    return selected, arrays, elapsed


def fit_projection(X_train, X_val, cfg):
    """Fit PCA exclusively to training features and transform validation features."""
    pca = PCA(n_components=cfg['pca_max_components'], svd_solver='randomized', iterated_power=cfg['pca_iterated_power'], random_state=cfg['seed'])
    start = perf_counter(); pca.fit(X_train); train_pca = pca.transform(X_train); fit_seconds = perf_counter() - start
    start = perf_counter(); val_pca = pca.transform(X_val); transform_seconds = perf_counter() - start
    return pca, train_pca, val_pca, fit_seconds, transform_seconds


def save_validation_visuals(exp_id, rows, pred):
    """Persist a confusion matrix and deterministic incorrect validation examples."""
    fig_dir = OUT / 'figures'; met_dir = OUT / 'metrics'
    cm = confusion_matrix(rows.label, pred, labels=range(102)); np.save(met_dir / f'{exp_id}_validation_confusion.npy', cm)
    fig, ax = plt.subplots(figsize=(10, 9)); im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set(title=f'{exp_id} validation confusion matrix', xlabel='Predicted class ID', ylabel='True class ID')
    ticks = list(range(0, 102, 10)); ax.set_xticks(ticks, [str(x+1) for x in ticks]); ax.set_yticks(ticks, [str(x+1) for x in ticks]); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(fig_dir / f'{exp_id}_validation_confusion.png', dpi=130); plt.close(fig)
    predictions = rows[['image_id', 'class_id', 'path']].copy(); predictions['predicted_class_id'] = pred + 1
    predictions.drop(columns='path').to_csv(met_dir / f'{exp_id}_validation_predictions.csv', index=False)
    wrong = predictions[predictions.class_id != predictions.predicted_class_id]
    wrong.groupby(['class_id', 'predicted_class_id']).size().sort_values(ascending=False).rename('count').to_csv(met_dir / f'{exp_id}_validation_confusions.csv')
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat: ax.axis('off')
    for ax, row in zip(axes.flat, wrong.head(12).itertuples()):
        with Image.open(row.path) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f'True {row.class_id}, predicted {row.predicted_class_id}')
    fig.suptitle(f'{exp_id}: validation errors'); fig.tight_layout(); fig.savefig(fig_dir / f'{exp_id}_validation_errors.png', dpi=120); plt.close(fig)


def run():
    """Reproduce V1 validation search and compare with train-only-fitted V2 candidates."""
    for folder in ('figures', 'metrics', 'models'): (OUT / folder).mkdir(parents=True, exist_ok=True)
    cfg = load_config(); old_cfg = v1_config(); df = load_dataset()
    print('Validated dataset:', df.groupby('split').size().to_dict(), flush=True)
    v1 = checked_v1_features(df, old_cfg)
    old_train = v1['splits'] == 'train'; old_val = v1['splits'] == 'val'
    baseline_records = json.loads((V1_MET / 'validation_search.json').read_text())
    baseline_winners = {}; baseline_all = []
    for exp_id, spec in old_cfg['experiments'].items():
        X = v1[spec['features']]; model, records = search(X[old_train], v1['labels'][old_train], X[old_val], v1['labels'][old_val], spec, old_cfg['seed'])
        previous = [r for r in baseline_records if r['experiment'] == exp_id]
        if len(records) != len(previous): raise ValueError(f'{exp_id} candidate count changed')
        for new, old in zip(records, previous):
            if new['params'] != old['params'] or not np.isclose(new['val_macro_f1'], old['val_macro_f1'], atol=1e-12, rtol=0) or not np.isclose(new['val_accuracy'], old['val_accuracy'], atol=1e-12, rtol=0):
                raise ValueError(f'{exp_id} did not reproduce V1 validation search: {new} vs {old}')
        winner = max(records, key=lambda r: (r['val_macro_f1'], r['val_accuracy']))
        baseline_winners[exp_id] = {'experiment': exp_id, 'variant': spec['features'], 'params': winner['params'], 'val_accuracy': winner['val_accuracy'], 'val_macro_f1': winner['val_macro_f1'], 'input_dimensions': X.shape[1], 'model_dimensions': X.shape[1], 'fit_seconds': winner['train_seconds'], 'val_predict_seconds': winner['val_predict_seconds'], 'pca_fit_seconds': 0.0, 'pca_val_transform_seconds': 0.0}
        baseline_all += [{'experiment': exp_id, **r} for r in records]
        print(f'{exp_id}: V1 validation reproduced', flush=True)
    # Retain only train and validation pixel features. V2 never predicts test.
    pixels = np.concatenate((v1['pixels'][old_train], v1['pixels'][old_val])); pixel_ids = np.concatenate((v1['ids'][old_train], v1['ids'][old_val])); del v1, X
    rows, arrays, extract_seconds = v2_features(df, cfg)
    train = arrays['splits'] == 'train'; val = arrays['splits'] == 'val'
    order = {int(image_id): i for i, image_id in enumerate(pixel_ids)}
    arrays['pixels'] = pixels[[order[int(i)] for i in arrays['ids']]]
    del pixels
    y_train = arrays['labels'][train]; y_val = arrays['labels'][val]; val_rows = rows[val]
    all_records, best = [], {}
    projections = {}
    for exp_id, spec in cfg['experiments'].items():
        for variant in spec['variants']:
            X_train, X_val = arrays[variant][train], arrays[variant][val]
            if spec['pca']:
                if variant not in projections:
                    projections[variant] = fit_projection(X_train, X_val, cfg)
                pca, train_pca, val_pca, pca_fit, pca_transform = projections[variant]
                options = [(n, train_pca[:, :n], val_pca[:, :n]) for n in cfg['pca_components']]
            else:
                pca = None; pca_fit = pca_transform = 0.0
                options = [(None, X_train, X_val)]
            for n, train_x, val_x in options:
                for params in cfg['knn']:
                    model = KNeighborsClassifier(**params, n_jobs=1)
                    start = perf_counter(); model.fit(train_x, y_train); fit_time = perf_counter()-start
                    start = perf_counter(); pred = model.predict(val_x); predict_time = perf_counter()-start
                    score = metrics(y_val, pred)
                    record = {'experiment': exp_id, 'variant': variant, 'pca_components': n, 'params': params, 'input_dimensions': int(X_train.shape[1]), 'model_dimensions': int(train_x.shape[1]), 'val_accuracy': score['accuracy'], 'val_macro_f1': score['macro_f1'], 'validation_metrics': score, 'fit_seconds': fit_time, 'val_predict_seconds': predict_time, 'pca_fit_seconds': pca_fit, 'pca_val_transform_seconds': pca_transform}
                    all_records.append(record)
                    if exp_id not in best or (record['val_macro_f1'], record['val_accuracy']) > (best[exp_id]['record']['val_macro_f1'], best[exp_id]['record']['val_accuracy']):
                        best[exp_id] = {'record': record, 'model': model, 'pca': pca, 'pred': pred.copy()}
            print(f'{exp_id} {variant} complete', flush=True)
    winners = list(baseline_winners.values()) + [best[e]['record'] for e in cfg['experiments']]
    overall = max(winners, key=lambda r: (r['val_macro_f1'], r['val_accuracy']))
    save_json(OUT / 'metrics/validation_search.json', all_records)
    save_json(OUT / 'metrics/baseline_reproduction.json', baseline_all)
    save_json(OUT / 'metrics/comparison.json', {'experiments': winners, 'selected_experiment': overall['experiment'], 'selection_rule': 'validation macro F1, then accuracy, then listed order', 'v2_feature_extraction_seconds_all_variants': extract_seconds, 'v1_feature_signature': json.loads((V1_PROC/'features.json').read_text())['signature'], 'v2_feature_signature': json.loads(V2_MANIFEST.read_text())['signature'], 'config_sha256': digest(ROOT / 'configs/experiments_v2.yaml')})
    pd.DataFrame(winners).to_csv(OUT / 'metrics/comparison.csv', index=False)
    for exp_id, info in best.items():
        save_validation_visuals(exp_id, val_rows, info['pred'])
        with (OUT / 'models' / f'{exp_id}.pkl').open('wb') as f:
            pickle.dump({'model': info['model'], 'pca': info['pca'], 'record': info['record'], 'config': cfg}, f)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([r['experiment'] for r in winners], [r['val_macro_f1'] for r in winners]); ax.set(ylabel='Validation macro F1', title='E1–E6 validation comparison'); fig.tight_layout(); fig.savefig(OUT/'figures/validation_comparison.png', dpi=150); plt.close(fig)
    print('Selected by validation:', overall['experiment'], overall['val_macro_f1'], flush=True)
    print('No V2 candidate was evaluated on test.', flush=True)

if __name__ == '__main__': run()
