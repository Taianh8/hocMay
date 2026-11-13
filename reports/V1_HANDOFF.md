# V1 handoff — PARTIAL

Implemented: official metadata and split validation, image integrity checks, deterministic preprocessing, RGB pixels and HOG, E1/E2/E3 validation search, train-only selected model, test metrics and confusion matrix, error plots, provenance checks, nine stage notebooks and targeted tests. Source is in `src/`; settings are in `configs/experiments.yaml`. Generated outputs belong in `results/metrics`, `results/figures`, and `results/models` and are ignored by Git.

Verified in the original environment: see final task response for executed commands. Full dataset, model execution, figures, metrics and notebook execution are unverified because network access and required Python dependencies were unavailable. There are no verified scores. This is not a V1 acceptance pass.

Decisions: use official split unchanged; select by validation macro F1, accuracy as tie-break; fit final selected estimator on train only; no learned feature transform or PCA. Class IDs in outputs remain official (1–102). Cached features are tied to image content and settings. Resizing distorts aspect ratios, and HOG loses color. The dependency ranges are not a lockfile; the feature cache tracks preprocessing and feature code plus relevant package versions; record the full environment for reproducibility.

Priorities: (1) obtain official data and install dependencies, execute tests and all notebooks, inspect output and fix failures; record package versions; (2) inspect EDA and errors and write evidence-backed conclusions; (3) investigate train-only scaling/PCA or aspect-preserving transforms using validation; (4) consider color-aware features or segmentation in V3, without tuning on test. Treat pickle artifacts as trusted local files only.
