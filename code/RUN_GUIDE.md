# Run guide

## Option A - Google Colab (recommended, free T4 GPU)
1. Runtime -> Change runtime type -> GPU.
2. `!git clone https://github.com/shhoryaaa/MUJ-DS-2430010318.git && cd MUJ-DS-2430010318/code`
3. `!pip install -q -r requirements.txt`
4. Kaggle credentials (once per session): upload `kaggle.json` to `/root/.kaggle/` or follow the `kagglehub` prompt.
5. `import kagglehub; path = kagglehub.dataset_download("nih-chest-xrays/sample"); print(path)`
6. `!python -m src.train --data-dir {path} --out-dir runs/exp1` (about 45 minutes on a T4)
7. Inference: `!python -m src.predict --model runs/exp1/final_best_model.keras --thresholds runs/exp1/optimized_thresholds.json --image some_xray.png`

## Option B - local machine
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
python -m unittest discover -s code/tests -v              # no GPU needed
```
Training needs a CUDA-capable GPU with a TensorFlow build matching your CUDA version.

## Expected outputs
`optimized_thresholds.json`, `test_report.json` (per-disease precision/recall/F1/AUC/AP and TP/FP/FN/TN), `final_best_model.keras`. Re-running will not reproduce the third decimal exactly because GPU kernels are not fully deterministic.

## Troubleshooting
| Symptom | Fix |
|:---|:---|
| `tf.function only supports singleton tf.Variables` | augmentation layers must be created once, outside the mapped function (already done in `data.py`) |
| Out-of-memory on the GPU | lower `BATCH_SIZE` in `src/config.py` |
| No metadata CSV found | pass the folder that contains `sample_labels.csv` |
| Very low F1 at 0.5 | expected; use the validation-tuned thresholds |
