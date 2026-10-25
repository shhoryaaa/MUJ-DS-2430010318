# Capstone documentation

Project: **Multi-Label Chest X-Ray Disease Classification Using EfficientNetV2, Transfer Learning, Fine-Tuning and Explainable AI** - Shhorya Agarwal (2430010318), Data Science, section D, Batch F.

| Quantity | Value |
|:---|---:|
| Images used / patients | 5,606 / 4,230 |
| Train / validation / test images | 3,931 / 849 / 826 (patient-level, zero overlap) |
| Final model | EfficientNetV2B0, fine-tuned, focal loss (Model 4) |
| Test macro ROC-AUC (14 diseases) | **0.7012** (micro ROC-AUC 0.7974) |
| Test macro F1 at threshold 0.5 -> validation-tuned thresholds | 0.0442 -> **0.1526** (micro F1 0.0848 -> 0.2101) |
| Gain over the from-scratch baseline CNN (macro AUC) | +0.1078 (0.5934 -> 0.7012) |
| Best / weakest disease by AUC | Edema (0.873) / Infiltration (0.547) |
| Innovation (CO5): two-backbone ensemble, 12-class macro AUC | 0.7312 versus 0.6955 for the single-model reference |
| Published reference (CheXNet, full dataset), mean AUROC | 0.8414 |

## Reading order

| # | Document | Content |
|:--|:---|:---|
| 1 | [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | formal report: abstract, related work, method, results, discussion, conclusion |
| 2 | [`METHODOLOGY.md`](METHODOLOGY.md) | every procedure, formula and algorithm used |
| 3 | [`EXPERIMENTS_AND_RESULTS.md`](EXPERIMENTS_AND_RESULTS.md) | all experiments with complete tables and figures |
| 4 | [`LIMITATIONS_AND_ETHICS.md`](LIMITATIONS_AND_ETHICS.md) | what the results do not show; ethical and safe-use notes |
| 5 | [`FUTURE_WORK.md`](FUTURE_WORK.md) | prioritised follow-up experiments |
| 6 | [`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md) | verifiable reproducibility items |

The detailed project write-up lives in [`../assignments/dl-project/README.md`](../assignments/dl-project/README.md); source code in [`../code/`](../code/README.md).

> **Medical disclaimer.** Educational research project. The model is not a medical device and must not be used for diagnosis or treatment decisions.
