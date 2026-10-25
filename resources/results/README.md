# Results (capstone mirror)

| File | Content |
|:---|:---|
| `test_metrics.json` | macro ROC-AUC 0.7012, micro ROC-AUC 0.7974, macro F1 0.0442, micro F1 0.0848 (threshold 0.5), macro precision 0.1721, macro recall 0.0271; 826 test images, 14 labels |
| `validation_model_comparison.csv` | best validation macro AUC and best epoch of the four models (kept separate from test results) |
| `test_per_disease_metrics.csv`, `per_disease_metrics_threshold_0.5.csv` | per-disease metrics at threshold 0.5 |
| `results.csv`, `final_per_disease_metrics.csv` | per-disease metrics at the validation-tuned thresholds (+ macro row) |

Full set and interpretation: [`../../assignments/dl-project/results/README.md`](../../assignments/dl-project/results/README.md).
