"""Plot cross-split duplicate candidates identified by the V4 audit."""
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from src.utils import ROOT


def main():
    out = ROOT/'results/v4'; rows = pd.read_csv(out/'duplicate_candidates.csv')
    if rows.empty:
        print('No candidate pairs to plot'); return
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 4*len(rows)), squeeze=False)
    for i, row in enumerate(rows.itertuples()):
        for j, (image_id, split) in enumerate(((row.image_id_a, row.split_a), (row.image_id_b, row.split_b))):
            with Image.open(ROOT/'data/raw/jpg'/f'image_{image_id:05d}.jpg') as im:
                axes[i,j].imshow(im.convert('RGB'))
            axes[i,j].set_title(f'{split} image {image_id}; pHash distance {row.phash_hamming}')
            axes[i,j].axis('off')
    fig.tight_layout(); fig.savefig(out/'duplicate_candidate_pairs.png', dpi=110); plt.close(fig)
    print(f'Saved contact sheet for {len(rows)} candidates')

if __name__ == '__main__': main()
