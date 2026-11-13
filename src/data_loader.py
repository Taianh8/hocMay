"""Read and validate the official dataset and split files."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from .utils import ROOT

EXPECTED = 8189

def load_dataset(raw: Path = ROOT / 'data/raw', inspect_images: bool = True) -> pd.DataFrame:
    """Return one row per official image; reject missing, extra or corrupt files."""
    raw = Path(raw)
    try:
        labels = np.asarray(loadmat(raw / 'imagelabels.mat')['labels']).ravel()
        split = loadmat(raw / 'setid.mat')
    except (OSError, KeyError) as exc:
        raise ValueError(f'Cannot read official MAT files in {raw}: {exc}. See README download instructions.') from exc
    if len(labels) != EXPECTED or not np.array_equal(np.unique(labels), np.arange(1, 103)):
        raise ValueError('Expected 8,189 labels and exactly class IDs 1–102.')
    ids = {}
    for name, key in [('train', 'trnid'), ('val', 'valid'), ('test', 'tstid')]:
        arr = np.asarray(split[key]).ravel()
        if not np.issubdtype(arr.dtype, np.integer) or len(np.unique(arr)) != len(arr):
            raise ValueError(f'{key} contains non-integer or duplicate image IDs')
        ids[name] = arr.astype(int)
    all_ids = np.concatenate(list(ids.values()))
    if len(all_ids) != EXPECTED or not np.array_equal(np.sort(all_ids), np.arange(1, EXPECTED + 1)):
        raise ValueError('Official splits must be disjoint and cover every image ID exactly once.')
    files = {p.name for p in (raw / 'jpg').glob('*') if p.is_file()}
    expected = {f'image_{i:05d}.jpg' for i in range(1, EXPECTED + 1)}
    if files != expected:
        raise ValueError(f'Image files mismatch: missing={sorted(expected-files)[:20]} ({len(expected-files)} total); unexpected={sorted(files-expected)[:20]} ({len(files-expected)} total)')
    rows = []
    errors = []
    for name, values in ids.items():
        for i in values:
            path = raw / 'jpg' / f'image_{i:05d}.jpg'
            row = {'image_id': int(i), 'class_id': int(labels[i-1]), 'label': int(labels[i-1]-1), 'split': name, 'path': str(path)}
            if inspect_images:
                try:
                    with Image.open(path) as im:
                        row.update(width=im.width, height=im.height, mode=im.mode)
                        im.verify()
                    with Image.open(path) as im:
                        im.load()
                except Exception as exc:
                    errors.append(f'{path.name}: {exc}')
            rows.append(row)
    if errors:
        raise ValueError(f'{len(errors)} invalid images; first errors: {errors[:20]}')
    return pd.DataFrame(rows).sort_values('image_id').reset_index(drop=True)
