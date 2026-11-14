# V4 handoff — QA and demo; E6 test pending

**Later cleanup note:** The LaTeX source is now absent; the project owner chose to keep it removed. The draft PDF remains a local ignored artifact. The rest of this document records the earlier V4 state.

V4 contains a read-only dataset and split audit, a frozen E6 manifest and inference path, a local Streamlit demo, integration tests, an academic LaTeX draft, and reproducibility guidance. V1–V3 configurations and historical metrics remain intact. No V2/V3/V4 final E6 test evaluation has been run. Dataset QA did inspect image contents from the official test split to detect duplicates; the V1 test result had also been observed previously. Therefore do not describe the official test as untouched or independent for the project as a whole.

## Data and leakage findings

The complete prior audit decoded all 8,189 RGB images, verified class IDs and the official disjoint/exhaustive 1,020/1,020/6,149 split, and examined 13,584,360 cross-split pairs. Two exact byte/decoded-RGB duplicate pairs exist: train 7307 ↔ test 7309, and validation 8077 ↔ test 8067. Six additional pHash candidates were found at 64-bit Hamming distance ≤4; one is train–validation and one train–test. All eight pairs and limitations are documented in `reports/V4_DATASET_AUDIT.md` and `results/v4/`. pHash is a screening method with unknown recall and false-positive rate; no sample was removed or relabeled. The resumed V4 check revalidated all image decoding, metadata, the candidate CSV checksum, archive and metadata hashes, and each of the eight recorded pairs; it did not repeat the complete perceptual search.

The code audit found deterministic per-image resize, HOG and HSV block normalization, no learned scaler, and PCA fit using the official training features in V2 or within each V3 training fold. E6 was selected by validation macro F1 after 56 V2 configurations; its advantage over E4 is small in V3 CV and selection optimism remains. V1 test feedback and identified cross-split duplicate content are material limitations. V4 makes no new causal or generalization claim.

## Frozen model and demo

`configs/e6_manifest_v1.json` freezes the original local E6: HOG 8 + HSV histogram, 128 randomized PCA components with seed 42, 3-neighbor cosine K-NN with distance weighting; train-only fitting on 1,020 images. It records split policy, feature settings, versions and SHA-256 hashes for source, data, model and historical results. `src/inference.py` verifies model/code hashes, core library versions, trained PCA/model configuration, class mapping and manifest preprocessing against the stored V2 training configuration before prediction. It reuses V2 preprocessing and features, never fits, and returns IDs 1–102. A hash detects accidental mismatch, not an attacker who can replace both pickle and manifest. Load only trusted local files; pickle is executable.

`app.py` accepts JPEG/PNG, limits input to 10 MiB and 20 MP, and returns the numeric ID with no confidence or vote-probability claim. `requirements-v4-lock.txt` records the observed environment, including Streamlit. The model file, caches, dataset and results are ignored by Git; retain or restore a matching trusted copy to run the demo. README explains regeneration in an isolated copy and new versioned manifests. GUI server was started on localhost and health returned `ok`; Streamlit AppTest verified the initial page and the upload/prediction branch using a mocked uploader with an official training image. Actual browser-based interaction was not verified because the integrated browser host was unavailable.

## Verification and deliverables

`python -m pytest -q` passed 18 tests after the resumed manifest-preprocessing check, including V1–V3 tests, saved E6 validation feature/prediction alignment, training-feature alignment, altered-hash and altered-preprocessing rejection, input rejection and the mocked Streamlit upload flow. `python -m scripts.check_v4_audit` passed. The previous CLI inference run on official train image 6765 returned class ID 1. `scripts.check_v2_artifacts` and `scripts.check_v3_artifacts` passed in the prior V4 session. `scripts.verify_v4_notebooks` executed report-only notebooks 09 and 10 and checked saved execution copies for V1 notebooks 00–08; it deliberately did not rerun V1 notebook 05, which would evaluate test. The TeX draft under `reports/latex/main.tex` has been compiled and inspected with actual figures; team identity and the E6 final-test section remain pending. The user has deprioritized further report work.

## Final evaluation readiness

E6 is frozen for a possible train-only evaluation with seed 42. Before any test prediction, explicitly approve and record the model artifact, training policy, metric definitions and final reporting plan; report all results without test-driven changes. Since the official test contains an exact train duplicate and has prior V1 exposure, disclose those limits and prefer a new independent, deduplicated holdout for stronger claims. A train+validation model would be a different protocol and cannot be silently substituted.

## Remaining work

- A human team should review the split-duplicate examples, report draft, authorship and course-specific rubric before submission.
- Verify the GUI in a real browser when one is available; AppTest does not replace manual browser upload verification.
- Preserve the trusted binary model and relevant ignored artifacts externally if sharing the repository.
- Do not run E6 final test without the user's explicit authorization.
