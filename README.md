# Oxford 102 Flowers — classical ML baseline

Classify the Oxford 102 Flowers images using the official split and three reproducible classical ML experiments. This is a university course foundation, not a completed academic report. **Status: partial; the dataset, full run and resulting scores have not yet been verified in this environment.**

## Data

Download `102flowers.tgz`, `imagelabels.mat`, and `setid.mat` from the [official Oxford 102 Flowers page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). Cite: M.-E. Nilsback and A. Zisserman, “Automated Flower Classification over a Large Number of Classes,” Indian Conference on Computer Vision, Graphics and Image Processing, 2008. Respect the dataset's published terms.

```bash
mkdir -p data/raw
curl -L https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -o data/raw/102flowers.tgz
curl -L https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -o data/raw/imagelabels.mat
curl -L https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat -o data/raw/setid.mat
tar -xzf data/raw/102flowers.tgz -C data/raw
```

The image files must be `data/raw/jpg/image_00001.jpg` through `image_08189.jpg`. The loader checks 8,189 labels, all 102 IDs, disjoint and exhaustive official splits, exact image names, and image decoding; it reports missing, unexpected or invalid files rather than silently skipping them. The archive itself is outside `jpg/` and is ignored by Git. Internal labels are 0–101; exported predictions use official IDs 1–102. The dataset does not include human-readable class names in these files.

## Setup and reproduce

Use Python 3.11 or 3.12; create an isolated environment and install the controlled dependency ranges:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline all
jupyter notebook
```

Run notebooks `00` through `08` in order from the project root or `notebooks/`. They share the same implementation as the CLI. Running `python -m src.pipeline all` produces the same stages without running the notebooks. Individual stages are `data`, `eda`, `preprocess`, `features`, `train`, `evaluate`, `errors`, `sanity`; training requires a feature cache and evaluation requires a saved selection. The CLI automatically builds or verifies the cache for training, evaluation and sanity. The first run reads and verifies every image, then extracts features; it may take significant time and disk space. The cache hashes image contents and settings to prevent stale reuse. Use a fresh environment and record resolved packages with `python -m pip freeze > results/metrics/environment.txt` when reporting experiments; dependency ranges are not a lockfile, so exact cross-version numerical reproducibility is not guaranteed.

## Experiments and methodology

| ID | Features | Model and validation search |
|---|---|---|
| E1 | RGB pixel vector | K-NN, k = 1, 3, 5 |
| E2 | grayscale HOG | K-NN, k = 1, 3, 5 |
| E3 | grayscale HOG | decision tree, max depth 10, 20, unlimited × min leaf 1, 5 |

Resize to 64×64 RGB with bilinear interpolation and normalize to [0,1]. Pixel vectors have 12,288 entries; HOG uses 9 orientations, 8×8 cells and 2×2 blocks (1,764 entries). Square resizing can distort images; HOG discards color. No learned preprocessing is used. No PCA is applied. Each candidate is fitted only on official train; select using validation macro F1, then accuracy, then config order. The selected train-fitted model is evaluated on official test; no retraining on validation and no test-based tuning. Treat repeated inspection of test results as a finite evaluation budget. Timings are machine-dependent. Parameters and seed are in `configs/experiments.yaml`.

## Files and results

`src/` contains shared loading, preprocessing, features, models, metrics, and staged orchestration. `notebooks/` documents each stage. `tests/` covers critical invariants. Generated data stays out of Git: `data/processed/features.npz` is a content-keyed cache; `results/metrics/` contains metadata, validation search, selection, test metrics, predictions and confusion counts; `results/figures/` contains EDA, preprocessing, confusion and error plots; `results/models/selected_model.pkl` stores the selected estimator. Pickle files must only be loaded from trusted local sources. The model and cache checksums bind artifacts to configuration and source content. No real metrics are currently available.

## Limitations and collaboration

No official class-name mapping is supplied here. Figure interpretation is deliberately left to inspection of actual outputs. The 102-class matrix can be hard to read; the pair CSV complements it. The test score estimates performance only on this official split, not deployment. The code does not automatically verify source authenticity against a published checksum. Dependency ranges are not fully pinned. To contribute, create a branch, add experiment IDs without changing existing meanings, keep all fitted transformations in the training path, add targeted tests, document validation selection, and review changes before merging. Do not commit raw images, caches, generated models or results containing large data. See `reports/V1_HANDOFF.md` and `AI_USAGE.md`.
