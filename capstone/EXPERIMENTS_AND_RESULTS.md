## Final Results

### Held-out test evaluation

The following metrics were calculated from the current Colab runtime's
`y_true_test`, `y_pred_prob_test`, and `y_pred_bin_test` arrays. They describe
the evaluated model represented by those arrays only. They are not assigned
to the other experiments.

| Evaluated test model | Test images | Test labels | Macro ROC-AUC | Micro ROC-AUC | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Model associated with the current Colab test arrays | 826 | 14 | 0.7012 | 0.7974 | 0.0442 | 0.0848 |

- **Macro ROC-AUC:** unweighted mean of the ROC-AUC for labels containing both classes.
- **Micro ROC-AUC:** ROC-AUC calculated over flattened label targets and probabilities.
- **Macro F1 and Micro F1:** calculated using the existing binary predictions produced by the notebook.
- **Test set size:** 826 images and 14 labels.
- **Prediction rule:** the notebook's existing `y_pred_bin_test` was used. This cell does not assume that its threshold is 0.5.

### Validation model comparison

These values are validation ROC-AUC and best epochs from `model_results`.
They are **not test results**.

| Model | Best validation ROC-AUC | Best epoch |
|---|---:|---:|
| Baseline CNN | 0.5837 | 5 |
| EfficientNetV2B0 — Transfer Learning | 0.6973 | 6 |
| EfficientNetV2B0 — Fine-Tuning | 0.7073 | 5 |
| Focal-Loss Model | 0.7097 | 4 |

### Interpretation and limitations

- The four model-validation scores must not be compared as if they were test scores.
- The current runtime supplied one set of test prediction arrays. Test metrics for other models must be calculated from their own held-out test predictions before those models can be included in a test comparison.
- This is a multi-label classification task. ROC-AUC and F1 measure different aspects of performance.
- Per-disease results and machine-readable metrics are stored under `resources/results/`.
- These results are for research and educational use, not clinical diagnosis.
