# Source code

Modular re-implementation of the notebook cells. The executed reference is the [Colab notebook](https://colab.research.google.com/drive/1jgplvn6oFYI93MF0KZumF6Mfa_XRWkIv); this package makes the pipeline reusable and testable.

| Module | Role | Notebook cells |
|:---|:---|:---|
| `src/config.py` | labels, image size, batch size, seed | 1, 3, 8 |
| `src/data.py` | metadata discovery, label parsing, **patient-level split**, `tf.data` pipeline with augmentation | 3, 6-9 |
| `src/models.py` | baseline CNN, EfficientNetV2B0 / DenseNet121 classifier, partial unfreezing, focal loss, asymmetric loss | 10, 12, 15, 17, ablation |
| `src/train.py` | staged training driver (frozen -> fine-tune -> focal) and threshold tuning | 13-20 |
| `src/evaluate.py` | threshold search, per-class report with confusion counts, Hanley-McNeil CI, macro AUC (framework-free) | 18-20, 25 |
| `src/gradcam.py` | Grad-CAM through the nested backbone and the head | 24 |
| `src/predict.py` | command-line inference with the saved model and thresholds | 28 |
| `tests/test_utils.py` | unit tests (no TensorFlow required) | - |

```
code/
├── README.md, RUN_GUIDE.md, requirements.txt
├── src/   (config, data, models, train, evaluate, gradcam, predict)
└── tests/ (test_utils.py)
```

**Status.** `evaluate.py` and the data split are covered by unit tests. The TensorFlow modules mirror cells that were executed in Colab but the refactored package itself has not been run end to end (see the reproducibility checklist).

Quick start: [`RUN_GUIDE.md`](RUN_GUIDE.md).
