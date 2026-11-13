"""Verify CV boundaries, saved predictions and validation statistics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from src.pipeline_v3 import OUT, V2, fixed_winners, load_config, official_metadata
from src.utils import ROOT, digest, save_json


def main():
    cfg = load_config(); fixed_winners(cfg); labels, ids = official_metadata()
    records = json.loads((OUT/'metrics/cv_folds.json').read_text())
    folds = json.loads((OUT/'metrics/folds.json').read_text())
    summary = json.loads((OUT/'metrics/cv_summary.json').read_text())
    assert summary['config_sha256'] == digest(ROOT/'configs/experiments_v3.yaml')
    assert summary['v2_comparison_sha256'] == digest(V2/'metrics/comparison.json')
    assert summary['v2_feature_sha256'] == digest(ROOT/'data/processed/features_v2.npz')
    train_ids = np.sort(ids['train']); forbidden = set(ids['val']) | set(ids['test'])
    assert len(folds) == cfg['cv_folds'] and len(records) == 25
    seen = []
    for fold in folds:
        train, held = np.array(fold['train_image_ids']), np.array(fold['holdout_image_ids'])
        assert np.array_equal(train_ids[fold['train_index']], train)
        assert np.array_equal(train_ids[fold['holdout_index']], held)
        assert not (set(train) & set(held)) and set(train) | set(held) == set(train_ids)
        assert not ((set(train)|set(held)) & forbidden)
        assert len(train) == 816 and len(held) == 204 and np.unique(labels[held-1]).size == 102
        seen.extend(held)
    assert sorted(seen) == list(train_ids)
    expected_runs = {(name, seed, fold) for name, seeds in [('E1',[None]), ('E4',[None]), ('E6',cfg['pca_seeds'])] for seed in seeds for fold in range(1,6)}
    assert {(r['experiment'],r['pca_seed'],r['fold']) for r in records} == expected_runs
    for r in records:
        held = np.array(folds[r['fold']-1]['holdout_image_ids']); pred = np.array(r['predictions'])
        assert len(pred) == len(held) and np.all((pred >= 0) & (pred < 102))
        score = f1_score(labels[held-1], pred, labels=range(102), average='macro', zero_division=0)
        assert abs(score-r['macro_f1']) < 1e-12
    for s in summary['summaries']:
        subset = [r['macro_f1'] for r in records if r['experiment']==s['experiment'] and r['pca_seed']==s['pca_seed']]
        assert np.allclose(subset, s['fold_macro_f1'], atol=1e-12, rtol=0)
        assert abs(np.mean(subset)-s['mean_macro_f1']) < 1e-12 and abs(np.std(subset)-s['std_macro_f1_population']) < 1e-12
    for exp in ('E4','E5','E6'):
        p = pd.read_csv(V2/'metrics'/f'{exp}_validation_predictions.csv')
        assert set(p.image_id) == set(ids['val']) and np.array_equal(p.class_id.to_numpy()-1, labels[p.image_id.to_numpy()-1])
        y, pred = p.class_id-1, p.predicted_class_id-1
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(102), zero_division=0)
        per = pd.read_csv(OUT/'metrics'/f'{exp}_per_class.csv')
        assert np.allclose(per.precision,precision) and np.allclose(per.recall,recall) and np.allclose(per.f1,f1) and np.array_equal(per.support,support)
        assert np.array_equal(confusion_matrix(y,pred,labels=range(102)), np.load(V2/'metrics'/f'{exp}_validation_confusion.npy'))
    paths = [ROOT/'configs/experiments_v3.yaml', ROOT/'configs/experiments_v2.yaml', ROOT/'data/raw/imagelabels.mat', ROOT/'data/raw/setid.mat', ROOT/'data/processed/features_v2.npz', V2/'metrics/comparison.json', ROOT/'src/pipeline_v3.py', ROOT/'src/preprocessing.py', ROOT/'src/features_v2.py'] + [V2/'metrics'/f'{e}_validation_predictions.csv' for e in ('E4','E5','E6')]
    save_json(OUT/'metrics/provenance.json', {str(p.relative_to(ROOT)): digest(p) for p in paths})
    print('V3 CV splits, predictions, summaries, validation metrics and provenance verified')

if __name__ == '__main__': main()
