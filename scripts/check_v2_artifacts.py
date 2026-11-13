"""Check persisted V2 models against saved validation predictions and metrics."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.evaluation import metrics
from src.utils import ROOT, digest


def main():
    out = ROOT / 'results/v2'
    comparison = json.loads((out/'metrics/comparison.json').read_text())
    manifest = json.loads((ROOT/'data/processed/features_v2.json').read_text())
    if manifest['signature'] != comparison['v2_feature_signature'] or digest(ROOT/'data/processed/features_v2.npz') != manifest['cache_sha256']:
        raise ValueError('V2 feature cache provenance mismatch')
    if digest(ROOT/'configs/experiments_v2.yaml') != comparison['config_sha256']:
        raise ValueError('V2 config checksum mismatch')
    with np.load(ROOT/'data/processed/features_v2.npz', allow_pickle=False) as f:
        splits, ids, labels = f['splits'], f['ids'], f['labels']
        val = splits == 'val'
        assert set(splits) == {'train', 'val'} and len(ids) == 2040
        for exp in ('E4', 'E5', 'E6'):
            with (out/'models'/f'{exp}.pkl').open('rb') as file: payload = pickle.load(file)  # Trusted local artifacts only.
            rec = payload['record']
            assert rec == next(r for r in comparison['experiments'] if r['experiment'] == exp)
            if exp == 'E5':
                with np.load(ROOT/'data/processed/features.npz', allow_pickle=False) as old:
                    old_val = old['splits'] == 'val'
                    assert np.array_equal(old['ids'][old_val], ids[val])
                    x = old['pixels'][old_val]
            else:
                x = f[rec['variant']][val]
            if payload['pca'] is not None: x = payload['pca'].transform(x)[:, :rec['pca_components']]
            pred = payload['model'].predict(x)
            saved = pd.read_csv(out/'metrics'/f'{exp}_validation_predictions.csv')
            assert np.array_equal(saved.image_id.to_numpy(), ids[val])
            assert np.array_equal(saved.class_id.to_numpy(), labels[val]+1)
            assert np.array_equal(saved.predicted_class_id.to_numpy(), pred+1)
            assert metrics(labels[val], pred) == rec['validation_metrics']
            assert np.array_equal(confusion_matrix(labels[val], pred, labels=range(102)), np.load(out/'metrics'/f'{exp}_validation_confusion.npy'))
            print(f'{exp}: model, validation predictions, metrics, and confusion matrix agree')
    best = max(comparison['experiments'], key=lambda r: (r['val_macro_f1'], r['val_accuracy']))
    assert comparison['selected_experiment'] == best['experiment']
    print('Validation selection and feature cache provenance agree')

if __name__ == '__main__': main()
