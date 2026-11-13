# Oxford 102 Flowers — classical ML baseline

Classify the Oxford 102 Flowers images using the official split and three reproducible classical ML experiments. This is a university course foundation, not a completed academic report. The results below are from the verified V1 run described in the handoff.

## Data

Download `102flowers.tgz`, `imagelabels.mat`, and `setid.mat` from the [official Oxford 102 Flowers page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). Cite: M.-E. Nilsback and A. Zisserman, “Automated Flower Classification over a Large Number of Classes,” Indian Conference on Computer Vision, Graphics and Image Processing, 2008. Respect the dataset's published terms.

```bash
mkdir -p data/raw
curl -L https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -o data/raw/102flowers.tgz
curl -L https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -o data/raw/imagelabels.mat
curl -L https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat -o data/raw/setid.mat
python - <<'PYCODE'
import hashlib
from pathlib import Path
checks = {'102flowers.tgz': '52808999861908f626f3c1f4e79d11fa', 'imagelabels.mat': 'e0620be6f572b9609742df49c70aed4d', 'setid.mat': 'a5357ecc9cb78c4bef273ce3793fc85c'}
for name, expected in checks.items():
    with (Path('data/raw') / name).open('rb') as source:
        actual = hashlib.file_digest(source, 'md5').hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch for {name}: {actual}')
    print(f'{name}: checksum OK')
PYCODE
tar -xzf data/raw/102flowers.tgz -C data/raw
```


In this run, metadata came directly from Oxford and the image archive came from [this mirror](https://huggingface.co/datasets/XiN0919/FGVC/blob/main/102flowers.tgz) because the Oxford archive download was very slow. The downloaded files matched the [torchvision dataset checksums](https://github.com/pytorch/vision/blob/main/torchvision/datasets/flowers102.py): `102flowers.tgz` MD5 `52808999861908f626f3c1f4e79d11fa`, `imagelabels.mat` MD5 `e0620be6f572b9609742df49c70aed4d`, and `setid.mat` MD5 `a5357ecc9cb78c4bef273ce3793fc85c`. If the Oxford archive is unavailable or too slow, use `curl -fL https://huggingface.co/datasets/XiN0919/FGVC/resolve/main/102flowers.tgz -o data/raw/102flowers.tgz`, then run the same checksum check before extraction.

The image files must be `data/raw/jpg/image_00001.jpg` through `image_08189.jpg`. The loader checks 8,189 labels, all 102 IDs, disjoint and exhaustive official splits, exact image names, and image decoding; it reports missing, unexpected or invalid files rather than silently skipping them. The archive itself is outside `jpg/` and is ignored by Git. Internal labels are 0–101; exported predictions use official IDs 1–102. The dataset does not include human-readable class names in these files.

## Setup and reproduce

Use Python 3.11 or 3.12; create an isolated environment and install the pinned versions recorded from Python 3.11.15 on macOS arm64:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python -m pytest -q
python -m src.pipeline all
python scripts/execute_notebooks.py
jupyter notebook
```

Run notebooks `00` through `08` in order from the project root or `notebooks/`. They share the same implementation as the CLI. Running `python -m src.pipeline all` produces the same stages without running the notebooks. Individual stages are `data`, `eda`, `preprocess`, `features`, `train`, `evaluate`, `errors`, `sanity`; training requires a feature cache and evaluation requires a saved selection. The CLI automatically builds or verifies the cache for training, evaluation and sanity. Notebook execution requires local loopback sockets for the Jupyter kernel; executed copies and a status log are written to `results/notebooks/`. The first run reads and verifies every image, then extracts features; it may take significant time and disk space. The cache hashes image contents and settings to prevent stale reuse. The full resolved dependency set is in `requirements-lock.txt`; wheels and floating-point behavior may differ across platforms. `requirements.txt` retains compatible ranges for future upgrades.

## Experiments and methodology

| ID | Features | Model and validation search |
|---|---|---|
| E1 | RGB pixel vector | K-NN, k = 1, 3, 5 |
| E2 | grayscale HOG | K-NN, k = 1, 3, 5 |
| E3 | grayscale HOG | decision tree, max depth 10, 20, unlimited × min leaf 1, 5 |

Resize to 64×64 RGB with bilinear interpolation and normalize to [0,1]. Pixel vectors have 12,288 entries; HOG uses 9 orientations, 8×8 cells and 2×2 blocks (1,764 entries). Square resizing can distort images; HOG discards color. No learned preprocessing is used. No PCA is applied. Each candidate is fitted only on official train; select using validation macro F1, then accuracy, then config order. The selected train-fitted model is evaluated on official test; no retraining on validation and no test-based tuning. Treat repeated inspection of test results as a finite evaluation budget. Timings are machine-dependent. Parameters and seed are in `configs/experiments.yaml`.

## Files and results

`src/` contains shared loading, preprocessing, features, models, metrics, and staged orchestration. `notebooks/` documents each stage. `tests/` covers critical invariants. Generated data stays out of Git: `data/processed/features.npz` is a content-keyed cache; `results/metrics/` contains metadata, validation search, selection, test metrics, predictions and confusion counts; `results/figures/` contains EDA, preprocessing, confusion and error plots; `results/models/selected_model.pkl` stores the selected estimator. Pickle files must only be loaded from trusted local sources. The model and cache checksums bind artifacts to configuration and source content. The run on Python 3.11.15/macOS arm64 produced the following validation results (best configuration per experiment):

| Experiment | Parameters | Accuracy | Macro F1 |
|---|---|---:|---:|
| E1 | k=1 | 0.1088 | 0.0914 |
| E2 | k=1 | 0.0608 | 0.0584 |
| E3 | unlimited depth, min leaf=1 | 0.0245 | 0.0230 |

Validation selected E1. On the 6,149-image test split, E1 achieved accuracy 0.1057, macro precision 0.1275, macro recall 0.1052, macro F1 0.0807, weighted precision 0.1621, weighted recall 0.1057 and weighted F1 0.0853. E2/E3 were not evaluated on test. Exact values and timing are in the generated JSON. HOG did not improve on the pixel baseline in this search. These modest scores and the class-specific confusion patterns call for further analysis, not a claim of practical readiness.

## Limitations and collaboration

No official class-name mapping is supplied here. Figure interpretation is deliberately left to inspection of actual outputs. The 102-class matrix can be hard to read; the pair CSV complements it. The test score estimates performance only on this official split, not deployment. The loader validates structure and decoding but does not automatically authenticate source files. The archive and metadata in this run were checked against torchvision MD5 values; MD5 is an integrity check, not a cryptographic authenticity guarantee. The pinned environment was verified only on macOS arm64. To contribute, create a branch, add experiment IDs without changing existing meanings, keep all fitted transformations in the training path, add targeted tests, document validation selection, and review changes before merging. Do not commit raw images, caches, generated models or results containing large data. See `reports/V1_HANDOFF.md` and `AI_USAGE.md`.

## V2: feature engineering and validation-only selection

V2 retains V1 configuration and artifacts, and uses `configs/experiments_v2.yaml` and `results/v2/` separately. It evaluates only the official train and validation splits for V2; no V2 model has been evaluated on test. The existing test result above is V1 E1 only. With the environment already set up as above, reproduce and verify:

```bash
python -m src.pipeline_v2
python -m scripts.check_v2_artifacts
python -m pytest -q
python scripts/verify_v2_notebooks.py v1
python scripts/verify_v2_notebooks.py v2
```

The verification runner routes V1 notebook outputs into `results/v2/notebook_v1_check/` so that original V1 artifacts are not overwritten; its executed notebook copies and execution records are in `results/v2/notebooks/`. Jupyter requires loopback sockets. The V2 feature cache `data/processed/features_v2.npz` contains train and validation only, with content/config/code/version signatures. V1 baseline search is recomputed and each candidate's validation score and parameters compared to saved V1 records; a mismatch aborts. The checksum check script loads only trusted local pickle files.

E4 concatenates independently L2-normalized grayscale HOG and RGB or HSV per-channel histograms (16 bins/channel), giving equal block weights. Two HOG cell sizes are tried: 8 and 16 pixels, with 9 orientations and 2×2 blocks. E5 uses normalized RGB pixels plus PCA. E6 applies PCA to each HOG + color variant. PCA is fit and centered using **train only**, without whitening or feature standardization. For efficiency, one randomized 128-component PCA is fitted per feature set (seed 42, 3 power iterations); the 64-component candidate uses the first 64 components of that fit. This need not equal fitting a separate randomized 64-component PCA. The four K-NN settings are listed explicitly in the YAML; the bounded V2 search has 56 candidates. Hyperparameters and model family are selected by validation macro F1, then accuracy, then configuration order; no test feedback is used. For a saved PCA model, use `pca.transform(X)[:, :record['pca_components']]` before K-NN prediction.

Verified validation results from this machine:

| ID | Best configuration | Dimensions input → model | Accuracy | Macro F1 | Fit s | Validation prediction s | PCA fit + train transform s | PCA validation transform s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | pixels, k=1 | 12288 → 12288 | 0.1088 | 0.0914 | 0.025 | 0.245 | — | — |
| E2 | HOG, k=1 | 1764 → 1764 | 0.0608 | 0.0584 | 0.001 | 0.029 | — | — |
| E3 | HOG, unlimited depth, min leaf=1 | 1764 → 1764 | 0.0245 | 0.0230 | 16.797 | 0.003 | — | — |
| E4 | HOG 8 + HSV, k=5, Euclidean, distance | 1812 → 1812 | 0.2186 | 0.2003 | 0.001 | 0.030 | — | — |
| E5 | pixels + 64 PCA, k=3, cosine, distance | 12288 → 64 | 0.1500 | 0.1334 | <0.001 | 0.008 | 0.201 | 0.011 |
| E6 | HOG 8 + HSV + 128 PCA, k=3, cosine, distance | 1812 → 128 | 0.2343 | 0.2127 | <0.001 | 0.008 | 0.028 | 0.002 |

Timings are machine-dependent. PCA times include a shared 128-component fit and are not per-candidate search costs; V2 feature extraction took 7.82 s for all four variants on 2,040 images, excluding cache hashing/I/O. V1 feature extraction time was not recorded, so these are not comparable end-to-end runtimes. Exact values and full candidate timings are in `results/v2/metrics/`. E6 improves validation macro F1 by 0.1212 absolute over E1; this is a validation result after searching configurations, not an estimate of test improvement. HSV variants outperformed their corresponding RGB variants in these searches; HOG cell size 8 versus 16 was nearly tied for the best HSV configurations. More analysis is needed before attributing individual errors to visual causes.

V2 plots and validation error examples are under `results/v2/figures/`; prediction tables, confusion counts and matrices under `results/v2/metrics/`; models under `results/v2/models/`. These generated files and large datasets remain ignored by Git. See `reports/V2_HANDOFF.md` for limitations and next steps.
