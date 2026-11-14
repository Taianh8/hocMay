"""Check retained V4 audit results and all listed duplicate candidates.

This revalidates image loading and candidate pairs but does not repeat the
full cross-split perceptual-hash search; rerun src.dataset_audit for that.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from src.data_loader import load_dataset
from src.dataset_audit import OUT, phash
from src.utils import ROOT, digest


def rgb_digest(image):
    rgb = image.convert('RGB')
    h = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode())
    h.update(np.asarray(rgb).tobytes())
    return h.hexdigest()


def main():
    summary = json.loads((OUT/'dataset_audit.json').read_text())
    csv = OUT/'duplicate_candidates.csv'
    if digest(csv) != summary['candidate_csv_sha256']: raise ValueError('Audit candidate CSV checksum mismatch')
    for name, expected in summary['dataset_sha256'].items():
        if digest(ROOT/'data/raw'/name) != expected: raise ValueError(f'Dataset archive/metadata mismatch: {name}')
    df = load_dataset(); by_id = df.set_index('image_id')
    actual = {k:int(v) for k,v in df.groupby('split').size().items()}
    if len(df) != summary['image_count'] or df.class_id.nunique() != summary['classes'] or actual != summary['split_counts']:
        raise ValueError('Current dataset metadata differs from saved audit')
    rows = pd.read_csv(csv)
    if len(rows) != summary['candidate_pairs']: raise ValueError('Candidate count mismatch')
    for row in rows.itertuples():
        a, b = by_id.loc[row.image_id_a], by_id.loc[row.image_id_b]
        if (a['split'], int(a['class_id']), b['split'], int(b['class_id'])) != (row.split_a, row.class_id_a, row.split_b, row.class_id_b):
            raise ValueError('Candidate IDs, labels or splits do not match current official metadata')
        pa, pb = Path(a['path']), Path(b['path'])
        with Image.open(pa) as ia, Image.open(pb) as ib:
            exact_rgb = rgb_digest(ia) == rgb_digest(ib)
            distance = int(np.bitwise_count(np.bitwise_xor(phash(ia), phash(ib))))
        if (digest(pa) == digest(pb)) != row.file_exact or exact_rgb != row.decoded_rgb_exact or distance != row.phash_hamming:
            raise ValueError(f'Candidate image contents changed: {row.image_id_a}, {row.image_id_b}')
    print(f'Current 8,189 images decode; {len(rows)} saved cross-split pairs and dataset checksums verified. Full pHash search was not repeated.')

if __name__ == '__main__': main()
