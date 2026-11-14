# Cleanup audit and deletion manifest

Scope: preserve the current working tree and Git history, E1–E6 configurations and results, V3 CV evidence, V4 duplicate audit, all source notebooks, V4 frozen E6 model and manifest, figures, dataset and documentation. No new official test evaluation is authorized by this cleanup.

| Decision | Paths or category | Reason |
|---|---|---|
| DELETE | `.pytest_cache/`, `scripts/__pycache__/`, `src/__pycache__/`, `tests/__pycache__/` | Rebuildable interpreter/test caches, not research evidence. |
| DELETE | `results/v4/main.pdf`, `results/v4/main.log` | Superseded TeX trial: PDF contains missing-figure placeholders; not the verified report. |
| DELETE | `results/v4/qa/main.pdf`, `results/v4/qa/main.log`, then empty `results/v4/qa/` | Earlier superseded TeX QA rendering; the verified four-page rendering and log remain in `results/v4/qa2/`. |
| KEEP | `data/raw/`, `data/processed/`, `.venv/` | Official data and checksummed caches are needed to reproduce and verify results; the isolated environment is useful locally. These are Git-ignored. |
| KEEP | `results/metrics/`, `results/v2/metrics/`, `results/v3/metrics/`, `results/v4/dataset_audit.json`, `results/v4/duplicate_candidates.csv` | Historical scores, predictions, fold membership, provenance and dataset audit evidence. |
| KEEP | `results/figures/`, `results/v2/figures/`, `results/v3/figures/`, `results/v4/duplicate_candidate_pairs.png` | Verified figures and confusion/error analysis; large/generated figures remain Git-ignored. |
| KEEP | `results/models/`, `results/v2/models/` | Historical fitted estimators, including frozen E6. Pickles are large/trusted-local-only and Git-ignored. |
| KEEP | `results/notebooks/`, `results/v2/notebooks/`, `results/v2/notebook_v1_check/`, `results/v3/notebooks/`, `results/v4/notebooks/` | Executed notebook evidence and isolated outputs used for verification; some original notebook outputs are already tracked. Although some may be reproducible, they are not assumed redundant. |
| KEEP | `results/v3/work/`, experiment logs and V1–V4 handoffs | Retained because their future value or provenance is not certain; generated/cache-like material remains ignored. |
| KEEP | `requirements.txt`, `requirements-lock.txt`, `requirements-v4-lock.txt` | Distinct purposes: compatible ranges, historical pinned environment, V4 environment including Streamlit. Do not merge locks. |
| REFACTOR | Notebook output text and local notebook execution log | Replace a workstation-specific project-root prefix with a relative path; preserve execution output and metrics. |
| REFACTOR | `.gitignore`, README and handoff documentation | Continue ignoring bulky or local generated files while allowing compact machine-readable results required for provenance to be tracked. |
| KEEP | All remaining source modules, scripts, configs, notebooks and historical handoffs | No script/module was proven obsolete; preserve research lineage, user work and reproducibility. |

Localhost in the Streamlit command is an intentional loopback binding, not a machine-specific filesystem path. The V4 LaTeX source disappeared during cleanup without an action by the assistant; the project owner subsequently chose to keep it removed. The verified draft rendering remains in `results/v4/qa2/`. Final E6 test results remain pending. A basic credential-pattern scan is not a security audit and cannot prove that no secret exists.
