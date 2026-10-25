# Assignments

| Assignment | Course | Topic | Outcome mapping | Folder |
|:---|:---|:---|:---|:---|
| Individual Deep Learning project (20 marks) | Deep Learning (DSE3120) | Multi-Label Chest X-Ray Disease Classification Using EfficientNetV2, Transfer Learning, Fine-Tuning and Explainable AI | CO3, CO4, CO5 | [`dl-project/`](dl-project/README.md) |

## Contents of `dl-project/`

* [`README.md`](dl-project/README.md) - the full project document (22 sections, tables, figures, per-disease analysis)
* [`CO_MAPPING.md`](dl-project/CO_MAPPING.md) - evidence for CO3, CO4 and CO5
* [`results/`](dl-project/results/README.md) - CSV outputs; [`figures/`](dl-project/figures/README.md) - 34 figures; [`models/`](dl-project/models/README.md) - thresholds, labels, description
* [`requirements.txt`](dl-project/requirements.txt) - versions of the packages of the run

## Headline result

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

> **Medical disclaimer.** Educational research project. The model is not a medical device and must not be used for diagnosis or treatment decisions.
