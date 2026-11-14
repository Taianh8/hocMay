# Team handoff — Oxford 102 Flowers

V1–V4 are implemented. V1 E1–E3 and V2 E4–E6 historical results, V3 train-only CV, V4 duplicate audit, 11 source notebooks and the frozen E6 demo remain in place. E6 validation macro F1 is 0.2127; E4 is 0.2003. V3 five-fold train-only macro F1 is 0.1932 ± 0.0195 for E6 seed 42 and 0.1852 ± 0.0148 for E4. The small difference and selection on the validation set limit generalization claims.

## Start here

- [README.md](README.md): dataset source, architecture, exact commands, results and limitations. [CONTRIBUTING.md](CONTRIBUTING.md) sets team rules.
- `configs/experiments*.yaml`: historical experiment settings; `configs/e6_manifest_v1.json`: frozen E6, version and input/model hashes. `src/inference.py` and `app.py` use the trusted local model.
- `results/metrics/`, `results/v2/metrics/`, `results/v3/metrics/` and `results/v4/` contain scores, CV and duplicate findings. Compact CSV/JSON is eligible for tracking; verify untracked files before any commit. Figures, data, notebook execution copies, caches and model binaries remain local and ignored.
- `reports/V1_HANDOFF.md` through `reports/V4_HANDOFF.md` preserve original evidence; `reports/V4_DATASET_AUDIT.md` lists all duplicate candidates. The LaTeX draft source is absent by the project owner’s decision. The previously rendered draft remains at `results/v4/qa2/main.pdf`. AI assistance is recorded in `AI_USAGE.md`.

## Reproduce and verify

Use Python 3.11 on the verified macOS arm64 platform; install `requirements-v4-lock.txt` in `.venv`. Wheels or exact numeric results may differ on other platforms. Obtain the official dataset and preserve its three original split metadata files. Restore the original **trusted** `results/v2/models/E6.pkl`, V2 feature cache and validation outputs that match the manifest; they are not supplied by Git. Pickle hashes detect accidental changes, not a malicious replacement of both model and manifest.

From the repository root, after setup, run `python -m pytest -q`, `python -m scripts.check_v4_audit`, and `python scripts/verify_v4_notebooks.py`. The notebook checker needs Jupyter loopback; it validates retained V1 execution copies and executes only report notebooks. To smoke-test inference without test data, use `python -m src.inference data/raw/jpg/image_06765.jpg`; launch the demo with `streamlit run app.py --server.address 127.0.0.1`. See README for the full setup and isolated reproduction procedures. Rebuilding V2/V3 or notebook 05 can overwrite historical outputs; notebook 05 and V1 `pipeline all` also evaluate test.

## Next contributor

1. Back up ignored datasets, figures, model and retained notebook outputs, and review compact CSV/JSON evidence before deciding what to track.
2. Verify Streamlit upload in a real browser; the recorded smoke test used AppTest with a mocked uploader. Review dataset candidate images, course rubric, authorship and any future report when requested; the removed LaTeX source should not be restored unless the owner asks.
3. Keep E6 final test **pending** until separately authorized. Freeze the exact artifact, train-only policy and reporting protocol before any test prediction; report all results without tuning. V1 test results were already inspected, and train image 7307 is byte-identical to test 7309; the official test is not an untouched independent holdout. Prefer an independent, deduplicated holdout for stronger claims.
