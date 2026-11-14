"""Create a versioned manifest for the existing trusted V2 E6 artifact."""
import argparse
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path
import numpy as np
from src.utils import ROOT, digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'configs/e6_manifest_v1.json', help='New manifest path; existing files are never overwritten')
    manifest_path = parser.parse_args().output
    if manifest_path.exists(): raise FileExistsError('Frozen manifest already exists; create a new version rather than overwriting it.')
    model_path = ROOT/'results/v2/models/E6.pkl'
    comparison_path = ROOT/'results/v2/metrics/comparison.json'
    comparison = json.loads(comparison_path.read_text())
    if comparison['selected_experiment'] != 'E6': raise ValueError('E6 is not the recorded V2 selection')
    with model_path.open('rb') as f: payload = pickle.load(f)  # existing trusted local model only
    record, pca, model = payload['record'], payload['pca'], payload['model']
    recorded = next(x for x in comparison['experiments'] if x['experiment']=='E6')
    if record != recorded or pca.random_state != 42 or pca.n_components_ != 128 or pca.n_samples_ != 1020 or model.n_samples_fit_ != 1020 or not np.array_equal(model.classes_, np.arange(102)):
        raise ValueError('E6 model, class mapping or training configuration does not match V2 selection')
    if digest(ROOT/'configs/experiments_v2.yaml') != comparison['config_sha256']: raise ValueError('V2 configuration checksum mismatch')
    files = ['configs/experiments_v2.yaml', 'src/preprocessing.py', 'src/features_v2.py', 'results/v2/metrics/comparison.json', 'results/v2/models/E6.pkl', 'data/raw/102flowers.tgz', 'data/raw/imagelabels.mat', 'data/raw/setid.mat', 'requirements-lock.txt', 'requirements-v4-lock.txt', 'data/processed/features_v2.npz', 'data/processed/features_v2.json', 'results/v2/metrics/E6_validation_predictions.csv', 'results/v3/metrics/cv_summary.json']
    versions = {p: importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','scikit-image','Pillow')}
    manifest = {'manifest_version': 1, 'experiment': 'E6', 'selection': {'validation_macro_f1':record['val_macro_f1'], 'validation_accuracy':record['val_accuracy'], 'rule':comparison['selection_rule'], 'final_test': 'pending; no V2/V3/V4 test evaluation'}, 'dataset': {'name':'Oxford 102 Flowers', 'images':8189, 'classes':102, 'splits':{'train':1020,'validation':1020,'test':6149}, 'mapping':'official class ID = internal predicted index + 1'}, 'preprocessing': {'mode':'RGB','size':[64,64], 'resampling':'Pillow bilinear', 'normalization':'float32 / 255', 'exif_transpose':False}, 'features': {'variant':'h8_hsv','orientations':9,'pixels_per_cell':8,'cells_per_block':2,'block_norm':'L2-Hys','histogram_bins_per_channel':16,'block_normalization':'each block L2 normalized; concatenated equal weight','input_dimensions':1812}, 'pca': {'components':128,'svd_solver':'randomized','iterated_power':3,'random_seed':42,'training_samples':1020}, 'knn': {'n_neighbors':3,'metric':'cosine','weights':'distance','n_jobs':1,'training_samples':1020}, 'python':platform.python_version(), 'dependencies':versions, 'complete_dependency_lock':'requirements-v4-lock.txt', 'sha256':{p:digest(ROOT/p) for p in files}, 'model_artifact':'results/v2/models/E6.pkl', 'trust': 'Local trusted pickle only; matching hashes detect accidental changes, not an attacker who can replace both manifest and model.'}
    save_json(manifest_path, manifest)
    print(f'Created {manifest_path}')

if __name__ == '__main__': main()
