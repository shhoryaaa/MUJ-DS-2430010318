# Reproducibility checklist

| Item | Status | Evidence |
|:---|:---|:---|
| Data source and version named | Done | Kaggle `nih-chest-xrays/sample`, version 4 |
| Random seeds fixed | Done | `SEED = 42` for Python, NumPy, TensorFlow; patient shuffle `RandomState(42)` |
| Patient-level split without overlap | Done | pairwise-disjoint assertion passed in the notebook and in [`code/tests`](../code/tests/test_utils.py) |
| Test set untouched during tuning | Mostly | thresholds and early stopping use validation only; the pick of Model 4 among four candidates used test macro AUC, but the validation ranking is identical |
| Software versions recorded | Done | Python 3.13.15, TensorFlow 2.21.0, [`code/requirements.txt`](../code/requirements.txt) |
| Hardware recorded | Done | Tesla T4, 15,360 MiB, mixed precision |
| Hyper-parameters documented | Done | [README section 6.3](../assignments/dl-project/README.md#63-hyper-parameters), [`METHODOLOGY.md`](METHODOLOGY.md) |
| Training logs stored | Done | `results/history_*.csv`, tables in the README |
| Raw metrics stored as data | Done | `results/*.csv`, `resources/results/*.json|csv` |
| Source code available | Done | [`code/src`](../code/README.md) (refactored from the notebook; the notebook is the executed reference) |
| Unit tests | Done | 4 tests for split, thresholds, CI and report counts |
| End-to-end re-run of `code/src/train.py` | Not done | the package was refactored after the executed run and has been unit-tested only; the Colab notebook remains the executed source of truth |
| Repeated runs / variance | Not done | single seed; listed in future work |
| Weights shared | Partly | `.keras` file excluded from git (size); delivered in the course ZIP |
