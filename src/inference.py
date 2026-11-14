"""Frozen E6 image inference; no fitting, tuning, or test-set evaluation."""
import argparse
from io import BytesIO
import importlib.metadata
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from .features_v2 import hog_color
from .preprocessing import preprocess
from .utils import ROOT, digest

MANIFEST = ROOT/'configs/e6_manifest_v1.json'
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def load_frozen_model(manifest_path: Path = MANIFEST):
    """Verify trusted manifest, environment and locally stored pickle before loading."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('manifest_version') != 1 or manifest.get('experiment') != 'E6':
        raise ValueError('Unsupported frozen model manifest')
    for package, expected in manifest['dependencies'].items():
        actual = importlib.metadata.version(package)
        if actual != expected: raise ValueError(f'{package} version mismatch: expected {expected}, got {actual}; install requirements-v4-lock.txt')
    for name in ('src/preprocessing.py', 'src/features_v2.py', manifest['model_artifact']):
        path = ROOT/name
        if not path.is_file() or digest(path) != manifest['sha256'][name]:
            raise ValueError(f'Missing or modified frozen artifact: {name}; restore the trusted V2 artifact and manifest')
    # Pickle can execute code: the manifest and model must both be trusted.
    with (ROOT/manifest['model_artifact']).open('rb') as f: payload = pickle.load(f)
    model, pca, record = payload['model'], payload['pca'], payload['record']
    pc = manifest['pca']; knn = manifest['knn']
    cfg = payload['config']
    variant = next((v for v in cfg['feature_variants'] if v['id'] == record['variant']), None)
    expected_preprocessing = {'mode': 'RGB', 'size': [cfg['image_size'], cfg['image_size']], 'resampling': 'Pillow bilinear', 'normalization': 'float32 / 255', 'exif_transpose': False}
    if variant is None:
        raise ValueError('Frozen model has no matching feature configuration')
    expected_features = {'variant': variant['id'], **cfg['hog_variants'][variant['hog']], 'block_norm': 'L2-Hys', 'histogram_bins_per_channel': cfg['histogram_bins'], 'block_normalization': 'each block L2 normalized; concatenated equal weight', 'input_dimensions': pca.n_features_in_}
    if (manifest['preprocessing'] != expected_preprocessing or manifest['features'] != expected_features
        or variant['color'] != 'hsv' or pc['random_seed'] != cfg['seed']
        or pc['iterated_power'] != cfg['pca_iterated_power'] or pc['components'] != cfg['pca_max_components']
        or manifest['dataset']['classes'] != 102 or manifest['dataset']['mapping'] != 'official class ID = internal predicted index + 1'):
        raise ValueError('Frozen manifest preprocessing or feature settings do not match the training configuration')
    if (record['experiment'] != 'E6' or record['variant'] != 'h8_hsv' or record['pca_components'] != pc['components']
        or pca.n_components_ != pc['components'] or pca.random_state != pc['random_seed'] or pca.svd_solver != pc['svd_solver']
        or pca.iterated_power != pc['iterated_power'] or pca.n_samples_ != pc['training_samples']
        or pca.n_features_in_ != manifest['features']['input_dimensions']
        or model.n_samples_fit_ != knn['training_samples'] or model.n_features_in_ != pc['components']
        or not np.array_equal(model.classes_, np.arange(102))
        or any(model.get_params()[key] != value for key, value in knn.items() if key != 'training_samples')):
        raise ValueError('Frozen model contents do not match manifest')
    return manifest, payload


def image_features(data: bytes, manifest: dict) -> np.ndarray:
    """Use the exact V2 RGB resize and HOG+HSV implementation for an uploaded image."""
    if not data or len(data) > MAX_BYTES: raise ValueError('Image must be nonempty and at most 10 MiB')
    try:
        with Image.open(BytesIO(data)) as im:
            if im.format not in ('JPEG', 'PNG'): raise ValueError('Only JPEG and PNG images are supported')
            if im.width * im.height > MAX_PIXELS: raise ValueError('Image exceeds the 20-megapixel limit')
            im.verify()
        image = preprocess(BytesIO(data), manifest['preprocessing']['size'][0])
        feat = manifest['features']; settings = {k:feat[k] for k in ('orientations','pixels_per_cell','cells_per_block')}
        vector = hog_color(image, settings, 'hsv', feat['histogram_bins_per_channel'])
        if vector.shape != (feat['input_dimensions'],) or not np.isfinite(vector).all(): raise ValueError('Invalid extracted feature vector')
        return vector
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f'Cannot decode image: {exc}') from exc


def predict_image(data: bytes, manifest: dict, payload: dict) -> int:
    """Return an official 1–102 class ID; no probability or confidence claim."""
    vector = image_features(data, manifest).reshape(1, -1)
    projected = payload['pca'].transform(vector)[:, :manifest['pca']['components']]
    prediction = np.asarray(payload['model'].predict(projected))
    if prediction.shape != (1,) or not 0 <= int(prediction[0]) < 102: raise ValueError('Invalid model prediction')
    return int(prediction[0]) + 1


def main():
    parser = argparse.ArgumentParser(description='Predict the official Oxford class ID using frozen E6.')
    parser.add_argument('image', type=Path)
    parser.add_argument('--manifest', type=Path, default=MANIFEST, help='Trusted versioned manifest path')
    args = parser.parse_args()
    manifest, payload = load_frozen_model(args.manifest)
    if args.image.stat().st_size > MAX_BYTES: raise ValueError('Image must be at most 10 MiB')
    print(json.dumps({'official_class_id': predict_image(args.image.read_bytes(), manifest, payload), 'experiment':'E6'}))

if __name__ == '__main__': main()
