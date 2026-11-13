"""V3: fixed-model, train-only CV and analysis of saved validation predictions.

No test predictions or test image reads are performed here. CV workers run in
separate processes for interpretable process-level peak RSS measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from .features_v2 import hog_color
from .preprocessing import preprocess
from .utils import ROOT, digest, save_json

OUT = ROOT / 'results/v3'
WORK = OUT / 'work'
V2 = ROOT / 'results/v2'


def load_config():
    """Load predeclared CV settings and the fixed historical winners."""
    with (ROOT / 'configs/experiments_v3.yaml').open() as f: return yaml.safe_load(f)


def fixed_winners(cfg):
    """Abort if V3 choices no longer match the historical V1/V2 winners."""
    data = json.loads((V2/'metrics/comparison.json').read_text())
    winners = {r['experiment']: r for r in data['experiments']}
    if data['selected_experiment'] != 'E6': raise ValueError('Historical V2 winner is not E6')
    for name in ('E1', 'E4', 'E6'):
        model, old = cfg['models'][name], winners[name]
        if model['feature'] != old['variant'] or model['params'] != old['params'] or model.get('pca_components') != old.get('pca_components'):
            raise ValueError(f'{name} configuration differs from the recorded V1/V2 winner')
    if digest(ROOT/'configs/experiments_v2.yaml') != data['config_sha256']:
        raise ValueError('V2 configuration differs from historical results')
    with (ROOT/'configs/experiments_v2.yaml').open() as f:
        v2_cfg = yaml.safe_load(f)
    if cfg['models']['E6']['pca_iterated_power'] != v2_cfg['pca_iterated_power'] or cfg['models']['E6']['pca_components'] != v2_cfg['pca_max_components'] or v2_cfg['seed'] not in cfg['pca_seeds']:
        raise ValueError('E6 PCA settings do not include the historical V2 configuration')
    return winners


def official_metadata():
    """Read and validate official split metadata, without reading test images."""
    raw = ROOT/'data/raw'
    labels = np.asarray(loadmat(raw/'imagelabels.mat')['labels']).ravel()
    sets = loadmat(raw/'setid.mat')
    if len(labels) != 8189 or not np.array_equal(np.unique(labels), np.arange(1, 103)):
        raise ValueError('Expected official 8,189 labels and classes 1–102')
    ids = {name: np.asarray(sets[key]).ravel().astype(int) for name, key in [('train','trnid'), ('val','valid'), ('test','tstid')]}
    all_ids = np.concatenate(list(ids.values()))
    if not np.array_equal(np.sort(all_ids), np.arange(1, 8190)) or [len(ids[k]) for k in ('train','val','test')] != [1020, 1020, 6149]:
        raise ValueError('Official split is not exhaustive, disjoint or expected size')
    return labels.astype(int)-1, ids


def validate_v2_cache(labels, ids):
    """Check V2 cache and label alignment; never use test features."""
    cache = ROOT/'data/processed/features_v2.npz'
    manifest = json.loads((ROOT/'data/processed/features_v2.json').read_text())
    comparison = json.loads((V2/'metrics/comparison.json').read_text())
    if digest(cache) != manifest['cache_sha256'] or manifest['signature'] != comparison['v2_feature_signature']:
        raise ValueError('V2 feature cache does not match historical artifacts')
    with np.load(cache, allow_pickle=False) as f:
        cache_ids, y, splits = f['ids'], f['labels'], f['splits']
        if set(splits) != {'train', 'val'} or len(cache_ids) != 2040 or len(set(cache_ids)) != 2040:
            raise ValueError('V2 cache contains unexpected or duplicate split records')
        for split in ('train', 'val'):
            selected = cache_ids[splits == split]
            if not np.array_equal(np.sort(selected), np.sort(ids[split])): raise ValueError(f'{split} IDs mismatch')
        if not np.array_equal(y, labels[cache_ids-1]): raise ValueError('V2 cache labels mismatch')
        return {int(i): vector for i, vector in zip(cache_ids[splits=='train'], f['h8_hsv'][splits=='train'])}


def feature_data(labels, ids, cache):
    """Re-extract only training images and compare HOG+HSV with V2 cache."""
    train_ids = np.sort(ids['train']); pix, combined = [], []
    settings = {'orientations': 9, 'pixels_per_cell': 8, 'cells_per_block': 2}
    preprocess_seconds = pixel_seconds = hog_color_seconds = 0.0
    for image_id in train_ids:
        path = ROOT/'data/raw/jpg'/f'image_{image_id:05d}.jpg'
        t = perf_counter(); image = preprocess(path, 64); preprocess_seconds += perf_counter()-t
        t = perf_counter(); p = image.ravel().copy(); pixel_seconds += perf_counter()-t
        t = perf_counter(); feat = hog_color(image, settings, 'hsv', 16); hog_color_seconds += perf_counter()-t
        if not np.array_equal(feat, cache[int(image_id)]):
            raise ValueError(f'V2 feature mismatch for training image {image_id}; do not proceed with stale artifacts')
        pix.append(p); combined.append(feat)
    np.save(WORK/'pixels.npy', np.stack(pix)); np.save(WORK/'h8_hsv.npy', np.stack(combined))
    np.save(WORK/'labels.npy', labels[train_ids-1]); np.save(WORK/'ids.npy', train_ids)
    return {'samples': len(train_ids), 'shared_decode_resize_normalize_seconds': preprocess_seconds, 'pixel_vector_seconds': pixel_seconds, 'hog_hsv_seconds': hog_color_seconds}


def make_folds(labels, ids, fold_count, seed):
    """Create identical stratified folds from official training IDs only."""
    y = labels[ids-1]
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    folds = []
    for train, holdout in splitter.split(np.zeros(len(ids)), y):
        if np.intersect1d(train, holdout).size or len(np.unique(y[holdout])) != 102:
            raise ValueError('Invalid or non-stratified CV fold')
        folds.append({'train_index': train.tolist(), 'holdout_index': holdout.tolist(), 'train_image_ids': ids[train].tolist(), 'holdout_image_ids': ids[holdout].tolist()})
    if sorted(i for fold in folds for i in fold['holdout_index']) != list(range(len(ids))):
        raise ValueError('CV holdout folds do not partition training samples')
    return folds


def run_fold(X, y, train_idx, hold_idx, model_config, seed):
    """Fit PCA and classifier on a single training fold, then score its holdout."""
    train_x, holdout_x = X[train_idx], X[hold_idx]
    pca_fit = pca_transform = 0.0
    if 'pca_components' in model_config:
        pca = PCA(n_components=model_config['pca_components'], svd_solver='randomized', iterated_power=model_config['pca_iterated_power'], random_state=seed)
        start = perf_counter(); pca.fit(train_x); train_x = pca.transform(train_x); pca_fit = perf_counter()-start
        start = perf_counter(); holdout_x = pca.transform(holdout_x); pca_transform = perf_counter()-start
    model = KNeighborsClassifier(**model_config['params'], n_jobs=1)
    start = perf_counter(); model.fit(train_x, y[train_idx]); fit_time = perf_counter()-start
    start = perf_counter(); pred = model.predict(holdout_x); pred_time = perf_counter()-start
    return {'macro_f1': float(f1_score(y[hold_idx], pred, labels=range(102), average='macro', zero_division=0)),
            'accuracy': float(np.mean(pred == y[hold_idx])), 'pca_fit_train_transform_seconds': pca_fit,
            'pca_holdout_transform_seconds': pca_transform, 'knn_fit_seconds': fit_time, 'knn_predict_seconds': pred_time,
            'input_dimensions': int(X.shape[1]), 'model_dimensions': int(train_x.shape[1]), 'predictions': pred.tolist()}


def worker(model_name, fold_number, seed):
    """Run a single isolated fold; report process-wide high-water RSS in bytes."""
    cfg = load_config(); model = cfg['models'][model_name]
    X = np.load(WORK/f"{model['feature']}.npy"); y = np.load(WORK/'labels.npy')
    folds = json.loads((OUT/'metrics/folds.json').read_text()); fold = folds[fold_number]
    result = run_fold(X, y, np.array(fold['train_index']), np.array(fold['holdout_index']), model, seed)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
    result.update(experiment=model_name, fold=fold_number+1, pca_seed=seed if model_name == 'E6' else None)
    print(json.dumps(result))


def validation_analysis(labels, ids):
    """Analyze only previously generated validation predictions, without refitting."""
    expected = np.sort(ids['val']); out = OUT/'metrics'; figdir = OUT/'figures'
    for exp in ('E4', 'E5', 'E6'):
        source = V2/'metrics'/f'{exp}_validation_predictions.csv'
        predictions = pd.read_csv(source).sort_values('image_id')
        image_ids = predictions.image_id.to_numpy(); y = predictions.class_id.to_numpy()-1; pred = predictions.predicted_class_id.to_numpy()-1
        if not np.array_equal(image_ids, expected) or not np.array_equal(y, labels[image_ids-1]) or not ((pred >= 0) & (pred < 102)).all():
            raise ValueError(f'{exp} predictions do not match official validation IDs and labels')
        cm = confusion_matrix(y, pred, labels=range(102))
        if not np.array_equal(cm, np.load(V2/'metrics'/f'{exp}_validation_confusion.npy')): raise ValueError(f'{exp} confusion matrix mismatch')
        winner = next(r for r in json.loads((V2/'metrics/comparison.json').read_text())['experiments'] if r['experiment'] == exp)
        observed_f1 = f1_score(y, pred, labels=range(102), average='macro', zero_division=0)
        if not np.isclose(observed_f1, winner['val_macro_f1'], atol=1e-12, rtol=0) or not np.isclose(np.mean(y == pred), winner['val_accuracy'], atol=1e-12, rtol=0):
            raise ValueError(f'{exp} predictions do not reproduce historical validation metrics')
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(102), zero_division=0)
        per_class = pd.DataFrame({'class_id': np.arange(1,103), 'precision': precision, 'recall': recall, 'f1': f1, 'support': support, 'correct': cm.diagonal(), 'predicted_count': cm.sum(axis=0)})
        per_class.to_csv(out/f'{exp}_per_class.csv', index=False)
        wrong = predictions[predictions.class_id != predictions.predicted_class_id]
        wrong.groupby(['class_id','predicted_class_id']).size().rename('count').reset_index().sort_values(['count','class_id','predicted_class_id'], ascending=[False, True, True]).to_csv(out/f'{exp}_confusion_pairs.csv', index=False)
        if exp == 'E6':
            fig, ax = plt.subplots(figsize=(15,4)); ax.bar(per_class.class_id, per_class.f1); ax.set(xlabel='Official class ID', ylabel='F1', title='E6 validation F1 by class (10 images per class)'); fig.tight_layout(); fig.savefig(figdir/'E6_per_class_f1.png', dpi=140); plt.close(fig)
            pairs = pd.read_csv(out/'E6_confusion_pairs.csv').head(4)
            examples = pd.concat([wrong[(wrong.class_id==r.class_id)&(wrong.predicted_class_id==r.predicted_class_id)].head(3) for r in pairs.itertuples()])
            examples.to_csv(out/'E6_top_confusion_examples.csv', index=False)
            fig, axes = plt.subplots(4, 3, figsize=(11, 12))
            for ax in axes.flat: ax.axis('off')
            for ax, row in zip(axes.flat, examples.itertuples()):
                with Image.open(ROOT/'data/raw/jpg'/f'image_{row.image_id:05d}.jpg') as image: ax.imshow(image.convert('RGB'))
                ax.set_title(f'{row.image_id}: {row.class_id} → {row.predicted_class_id}')
            fig.suptitle('E6 validation: examples from frequent directed confusions'); fig.tight_layout(); fig.savefig(figdir/'E6_confusion_examples.png', dpi=140); plt.close(fig)


def run():
    """Perform predeclared CV and validation error analysis; never evaluate test."""
    for d in (WORK, OUT/'metrics', OUT/'figures'): d.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); fixed_winners(cfg); labels, ids = official_metadata()
    cache = validate_v2_cache(labels, ids); extraction = feature_data(labels, ids, cache)
    train_ids = np.load(WORK/'ids.npy'); folds = make_folds(labels, train_ids, cfg['cv_folds'], cfg['fold_seed'])
    assert not (set(train_ids) & set(ids['val'])) and not (set(train_ids) & set(ids['test']))
    save_json(OUT/'metrics/folds.json', folds); save_json(OUT/'metrics/extraction.json', extraction)
    records = []
    env = os.environ.copy(); env['OPENBLAS_NUM_THREADS'] = '1'; env['OMP_NUM_THREADS'] = '1'
    for model in ('E1', 'E4', 'E6'):
        seeds = cfg['pca_seeds'] if model == 'E6' else [42]
        for seed in seeds:
            for fold_number in range(cfg['cv_folds']):
                cmd = [sys.executable, '-m', 'src.pipeline_v3', '--worker', model, str(fold_number), str(seed)]
                res = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT, env=env)
                record = json.loads(res.stdout); records.append(record)
                print(f"{model} fold={fold_number+1} pca_seed={record['pca_seed']} f1={record['macro_f1']:.6f}", flush=True)
    save_json(OUT/'metrics/cv_folds.json', records)
    summaries = []
    for model in ('E1','E4','E6'):
        seeds = cfg['pca_seeds'] if model == 'E6' else [None]
        for seed in seeds:
            sub = [r for r in records if r['experiment']==model and r['pca_seed']==seed]
            values = np.array([r['macro_f1'] for r in sub])
            summaries.append({'experiment': model, 'pca_seed': seed, 'fold_macro_f1': values.tolist(), 'mean_macro_f1': float(values.mean()), 'std_macro_f1_population': float(values.std(ddof=0)), 'mean_pca_fit_train_transform_seconds': float(np.mean([r['pca_fit_train_transform_seconds'] for r in sub])), 'mean_pca_holdout_transform_seconds': float(np.mean([r['pca_holdout_transform_seconds'] for r in sub])), 'mean_knn_fit_seconds': float(np.mean([r['knn_fit_seconds'] for r in sub])), 'mean_knn_predict_seconds': float(np.mean([r['knn_predict_seconds'] for r in sub])), 'mean_peak_rss_bytes': float(np.mean([r['peak_rss_bytes'] for r in sub])), 'input_dimensions': sub[0]['input_dimensions'], 'model_dimensions': sub[0]['model_dimensions']})
    save_json(OUT/'metrics/cv_summary.json', {'folds': cfg['cv_folds'], 'fold_seed': cfg['fold_seed'], 'summaries': summaries, 'config_sha256': digest(ROOT/'configs/experiments_v3.yaml'), 'v2_comparison_sha256': digest(V2/'metrics/comparison.json'), 'v2_feature_sha256': digest(ROOT/'data/processed/features_v2.npz'), 'train_ids_sha256': hashlib.sha256(train_ids.tobytes()).hexdigest()})
    validation_analysis(labels, ids)
    fig, ax = plt.subplots(figsize=(8, 4)); base = [r for r in summaries if r['pca_seed'] in (None,42)]; ax.bar([r['experiment'] for r in base], [r['mean_macro_f1'] for r in base], yerr=[r['std_macro_f1_population'] for r in base], capsize=5); ax.set(ylabel='Train-only CV macro F1', title='Five shared folds (E6 PCA seed 42)'); fig.tight_layout(); fig.savefig(OUT/'figures/cv_comparison.png', dpi=150); plt.close(fig)
    print('Completed: CV uses train only; validation analysis uses saved predictions; no test evaluations.', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', nargs=3, metavar=('MODEL','FOLD','SEED')); args = parser.parse_args()
    if args.worker: worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    else: run()

if __name__ == '__main__': main()
