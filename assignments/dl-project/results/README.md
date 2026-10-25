# Results folder

Machine-readable outputs of the experiments (all produced by the notebook run; derived files are marked). Human-readable interpretation: [`../README.md`](../README.md).

| File | Content |
| :--- | :--- |
| `results.csv` | per-disease accuracy, precision, recall, specificity, F1, ROC-AUC at the tuned thresholds (+ macro row) |
| `final_per_disease_metrics.csv` | same table written by the final-model cell |
| `per_disease_metrics_threshold_0.5.csv` | per-disease precision, recall, F1, ROC-AUC and average precision at the default threshold |
| `optimized_thresholds.csv` | validation-selected threshold of each disease |
| `model_comparison.csv / final_model_comparison.csv` | the four models on the test set (accuracy, macro precision/recall/F1, micro F1, macro AUC, training time) |
| `history_baseline.csv, history_transfer_learning.csv, history_fine_tuning.csv, history_imbalance_aware.csv` | per-epoch training/validation loss and AUC of each stage |
| `class_distribution.csv / split_distribution.csv` | positives per disease and split, prevalence, negative:positive ratio; images and patients per split |
| `error_analysis.csv` | false positives, false negatives and high-confidence mistakes per image/disease |
| `project_summary.csv` | one-line summary of the final model |
| `ablation.csv` | CO5 ablation (reference, +A, +A+B, ensemble) |
| `comparison.csv` | CO4 table: own models and published means on the same 12 classes (rewritten in the final update; no blanks) |
| `sota_comparison.csv, sota_per_class_auroc.csv` | published vs own AUROC, summary and per class |
| `per_disease_confusion_counts.csv` | TP/FP/FN/TN per disease at the tuned thresholds (derived exactly from the recorded metrics) |
| `per_disease_auc_ci.csv` | ROC-AUC with approximate 95% CI, average precision and lift over random |

The same test-set CSVs are mirrored for the capstone submission in [`../../../resources/results/`](../../../resources/results/README.md).
