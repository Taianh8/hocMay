"""Read-only Oxford data integrity and cross-split duplicate candidate audit."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from .data_loader import load_dataset
from .utils import ROOT, digest, save_json

OUT = ROOT/'results/v4'


def phash(image: Image.Image) -> np.uint64:
    """64-bit perceptual DCT hash; median of 63 low frequencies excludes DC."""
    gray = image.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(gray, dtype=np.float64), norm='ortho')[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    return np.uint64(sum(int(bit) << i for i, bit in enumerate(bits)))


def audit(threshold: int = 4):
    """Inspect every image and report all cross-split hash candidate pairs."""
    if not 0 <= threshold <= 64: raise ValueError('pHash threshold must be in [0,64]')
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    records = []
    for row in df.itertuples():
        path = Path(row.path)
        with Image.open(path) as im:
            rgb = im.convert('RGB'); raw = np.asarray(rgb)
            h = hashlib.sha256(); h.update(f'{rgb.width}x{rgb.height}:RGB:'.encode()); h.update(raw.tobytes())
            records.append({'image_id': row.image_id, 'split': row.split, 'class_id': row.class_id, 'file_sha256': digest(path), 'rgb_sha256': h.hexdigest(), 'phash': phash(rgb)})
    hashes = pd.DataFrame(records); pairs = []
    for a, b in combinations(('train', 'val', 'test'), 2):
        left = hashes[hashes.split==a].reset_index(drop=True); right = hashes[hashes.split==b].reset_index(drop=True)
        right_bits = right.phash.to_numpy(dtype=np.uint64)
        # Include exact hashes even when a perceptual implementation would differ.
        file_map, rgb_map = {}, {}
        for j, r in enumerate(right.itertuples()):
            file_map.setdefault(r.file_sha256, []).append(j); rgb_map.setdefault(r.rgb_sha256, []).append(j)
        for r in left.itertuples():
            distances = np.bitwise_count(np.bitwise_xor(right_bits, r.phash))
            candidates = set(np.flatnonzero(distances <= threshold).tolist()) | set(file_map.get(r.file_sha256, [])) | set(rgb_map.get(r.rgb_sha256, []))
            for j in sorted(candidates):
                other = right.iloc[j]; same_file = r.file_sha256 == other.file_sha256; same_pixels = r.rgb_sha256 == other.rgb_sha256
                pairs.append({'split_a': a, 'image_id_a': int(r.image_id), 'class_id_a': int(r.class_id), 'split_b': b, 'image_id_b': int(other.image_id), 'class_id_b': int(other.class_id), 'file_exact': bool(same_file), 'decoded_rgb_exact': bool(same_pixels), 'phash_hamming': int(distances[j])})
    columns = ['split_a','image_id_a','class_id_a','split_b','image_id_b','class_id_b','file_exact','decoded_rgb_exact','phash_hamming']
    candidates = pd.DataFrame(pairs, columns=columns).sort_values(['phash_hamming','image_id_a','image_id_b'])
    candidates.to_csv(OUT/'duplicate_candidates.csv', index=False)
    per_pair = candidates.groupby(['split_a','split_b']).size().to_dict()
    summary = {'image_count': len(df), 'classes': int(df.class_id.nunique()), 'split_counts': {k:int(v) for k,v in df.groupby('split').size().items()}, 'image_modes': {k:int(v) for k,v in df['mode'].value_counts().items()}, 'files_and_decoding': 'all expected image files decoded and verified', 'method': 'SHA-256 of file bytes and decoded RGB pixels (dimensions included); 64-bit DCT pHash of grayscale 32x32, median of 63 non-DC low frequencies', 'phash_hamming_threshold': threshold, 'cross_split_pairs_considered': int(sum(len(hashes[hashes.split==a])*len(hashes[hashes.split==b]) for a,b in combinations(('train','val','test'),2))), 'candidate_pairs': len(candidates), 'candidate_pairs_by_split': {f'{a}-{b}': int(n) for (a,b),n in per_pair.items()}, 'exact_file_pairs': int(candidates.file_exact.sum()), 'exact_decoded_rgb_pairs': int(candidates.decoded_rgb_exact.sum()), 'different_label_pairs': int((candidates.class_id_a != candidates.class_id_b).sum()), 'dataset_sha256': {name: digest(ROOT/'data/raw'/name) for name in ('102flowers.tgz','imagelabels.mat','setid.mat')}, 'candidate_csv_sha256': digest(OUT/'duplicate_candidates.csv')}
    save_json(OUT/'dataset_audit.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': audit()
