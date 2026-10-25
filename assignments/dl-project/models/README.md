# Model artefacts

| File | Content |
| :--- | :--- |
| `model_description.txt` | architecture, parameters, input/output contract and status of the final model |
| `optimized_thresholds.json` | per-disease thresholds selected on the validation split |
| `disease_labels.json` | ordered list of the 14 output labels |
| `final_best_model.keras` (not in git) | Keras weights of Model 4; excluded by `.gitignore` because of size - delivered in the ZIP and re-creatable with `code/src/train.py` |

Validation-selected thresholds:

| Disease | Threshold |
| :--- | ---: |
| Atelectasis | 0.11 |
| Cardiomegaly | 0.09 |
| Effusion | 0.31 |
| Infiltration | 0.13 |
| Mass | 0.11 |
| Nodule | 0.09 |
| Pneumonia | 0.05 |
| Pneumothorax | 0.23 |
| Consolidation | 0.05 |
| Edema | 0.21 |
| Emphysema | 0.11 |
| Fibrosis | 0.05 |
| Pleural_Thickening | 0.07 |
| Hernia | 0.05 |

> **Medical disclaimer.** Educational research project. The model is not a medical device and must not be used for diagnosis or treatment decisions.
