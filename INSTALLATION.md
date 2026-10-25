# Installation guide

1. **Clone:** `git clone https://github.com/shhoryaaa/MUJ-DS-2430010318.git`
2. **Environment:** Python 3.10 or newer; `pip install -r code/requirements.txt` (TensorFlow 2.x; a GPU is optional for evaluation, required in practice for training).
3. **Tests:** `python -m unittest discover -s code/tests -v` (no GPU, no dataset needed).
4. **Data:** create a Kaggle account, accept the NIH Chest X-rays terms, run `kagglehub.dataset_download("nih-chest-xrays/sample")`.
5. **Train / evaluate:** follow [`code/RUN_GUIDE.md`](code/RUN_GUIDE.md).
6. **Read the results without running anything:** [`assignments/dl-project/README.md`](assignments/dl-project/README.md).

Hardware: 8 GB RAM minimum (12-16 GB recommended), about 6 GB free disk for the sample dataset; Colab T4 GPU used for the reported run.

> Educational research project - not a medical device.
