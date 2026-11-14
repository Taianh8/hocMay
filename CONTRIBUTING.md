# Contributing

Use a branch and a Python 3.11 virtual environment with `requirements-v4-lock.txt`; the verified lock is macOS arm64 and another platform may need compatible wheels. Never commit raw data, generated binaries, caches or model pickles. Compact CSV/JSON result evidence can be tracked after review; preserve existing scores and provenance. Preserve official dataset splits and experiment IDs E1–E6.

Keep V1–V3 artifacts and historical scores unchanged. Learned transforms belong inside the training fold; use validation for development and never select or tune with the official test. Notebook 05 and `python -m src.pipeline all` perform the historical V1 test evaluation; do not rerun them as routine QA. `python -m src.pipeline_v2` regenerates V2 outputs, so run it in an isolated copy if the historical outputs must be preserved. Ask the project owner before any new final E6 test evaluation.

Before submitting, run `python -m pytest -q`, `python -m scripts.check_v4_audit`, and `python -m scripts.verify_v4_notebooks` where the relevant local artifacts and loopback access are available. The V4 integration tests depend on the trusted frozen model, feature cache and official data. Jupyter needs local loopback. Explain reproducibility, observed results and limitations, and update `AI_USAGE.md` for assistance used.

The model is a Python pickle: only load the original trusted local file and manifest. A matching local hash detects accidental drift but does not authenticate a maliciously replaced manifest and model. Changing trained model or preprocessing requires a new, reviewed versioned manifest and a separate documented evaluation protocol; never overwrite the frozen one.
