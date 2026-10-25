# Notebooks

| Notebook | Purpose | Link |
|:---|:---|:---|
| Multi-label chest X-ray classification (Colab, 34 cells) | complete executed pipeline: data -> models -> evaluation -> Grad-CAM -> ablation -> packaging | [Colab](https://colab.research.google.com/drive/1jgplvn6oFYI93MF0KZumF6Mfa_XRWkIv) |

Files stored in this folder: this README (the notebook itself is hosted on Colab).

Cell map: 1 environment, 2-3 data download and structure detection, 4-5 exploration, 6-7 cleaning and patient split, 8-9 pipeline and augmentation, 10-11 baseline CNN, 12-14 transfer learning, 15-16 fine-tuning, 17 focal loss, 18-20 comparison / per-disease / thresholds, 21-23 confusion, ROC, PR, 24 Grad-CAM, 25 errors, 26 TTA, 27 calibration, 28 prediction function, 29-31 dashboard, saving, summary, then CO5 ablation, packaging and repository upload.

The notebook is stored as a printed PDF in the course submission; an `.ipynb` copy is added to this folder when present.
