# V4 dataset integrity and duplicate audit

The retained audit validated all 8,189 expected image files and decoded them as RGB, 102 classes, and the disjoint, exhaustive official 1,020 train / 1,020 validation / 6,149 test ID split. `imagelabels.mat` maps image ID `i` to entry `i-1`; the internal index is the official class ID minus one. The source archive and two metadata files matched the SHA-256 values recorded in `configs/e6_manifest_v1.json`; their MD5 values also matched the previously checked torchvision metadata. No samples or splits were altered.

## Cross-split duplicate search

The audit compared **13,584,360** train–validation, train–test and validation–test pairs. Exact matching uses SHA-256 of JPEG bytes and separately of decoded RGB bytes with width and height included. Near-duplicate candidates use a 64-bit DCT perceptual hash: convert to grayscale, resize to 32×32, use the upper-left 8×8 DCT coefficients and threshold against the median of the 63 non-DC coefficients; report Hamming distance ≤ 4. The threshold was chosen for screening, not calibrated against labeled duplicate pairs. Exact hash matches are included regardless of pHash distance. A pHash match is a candidate, not proof of identical bytes or a statistically measured duplicate rate.

Eight cross-split candidate pairs were found; all share the same class label. Two are byte-identical and decoded-RGB-identical:

| Split A | Image A | Split B | Image B | Class | pHash distance | Byte-identical |
|---|---:|---|---:|---:|---:|---|
| validation | 1835 | test | 1858 | 78 | 0 | no |
| validation | 7015 | test | 7040 | 39 | 0 | no |
| validation | 7029 | test | 8080 | 39 | 0 | no |
| train | 7307 | test | 7309 | 94 | 0 | yes |
| validation | 8077 | test | 8067 | 31 | 0 | yes |
| validation | 1929 | test | 1959 | 78 | 2 | no |
| train | 1855 | test | 1844 | 78 | 4 | no |
| train | 5746 | validation | 5768 | 13 | 4 | no |

A contact sheet is in `results/v4/duplicate_candidate_pairs.png`; the full machine-readable list and summary are `results/v4/duplicate_candidates.csv` and `results/v4/dataset_audit.json`. The prior visual review showed closely matching compositions for the listed perceptual candidates; it does not establish whether they came from the same original capture. pHash can miss re-framing, rotation, reflections, color/lighting changes and other related images, and can produce false positives. Within-split duplicates and group-level photographic similarity were not exhaustively assessed. The true prevalence of related images and the influence on any metric are unknown.

The exact train–test pair is direct overlap between official partitions; the train–validation candidate and validation–test overlaps are additional risks for selection and generalization. These findings do not justify silently removing observations or changing the official split. Report the official result with this limitation and seek an independent deduplicated evaluation set for stronger claims. The audit inspected image contents across splits but did not run a new model evaluation on test.

`python -m scripts.check_v4_audit` revalidates decoding, metadata and the eight recorded candidates plus archive/metadata checksums. It intentionally does not repeat the full pair search or compare every extracted JPEG with the archive. To rerun the full search when warranted, use `python -m src.dataset_audit`, then `python -m scripts.duplicate_contact_sheet`; these overwrite the V4 audit outputs, so preserve a copy first if retaining historical output matters.
