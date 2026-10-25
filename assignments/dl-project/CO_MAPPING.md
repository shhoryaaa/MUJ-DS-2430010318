# Course-outcome evidence map (CO3, CO4, CO5)

Student: **Shhorya Agarwal** - Reg. No. **2430010318** - Branch Data Science, section D.
Marks as stated in the project brief: CO3 apply / implement = 10, CO4 state-of-the-art comparison = 5, CO5 innovation = 5 (total 20).

## CO3 - Apply: implement a deep-learning solution on real data (10 marks)

| Expected evidence | Delivered | Location |
|:---|:---|:---|
| Real, non-toy dataset with documented provenance | NIH ChestX-ray14 Kaggle sample, 5,606 images, 4,230 patients, 14 labels | [README section 3](README.md#3-dataset-and-exploratory-analysis) |
| Correct data handling, no leakage | patient-level 70/15/15 split with disjointness assertion; integrity pass (0 corrupt, 0 duplicate) | [section 4](README.md#4-data-preparation-and-patient-level-split) |
| Pre-processing and augmentation | 224 x 224 `tf.data` pipeline; mild medical-safe augmentation, no flips | [section 5](README.md#5-preprocessing-and-augmentation) |
| Baseline and pre-trained model | scratch CNN (95,950 params) and EfficientNetV2B0 (5.94 M params) | [section 6](README.md#6-models-and-architecture) |
| Transfer learning and fine-tuning | frozen head stage, then 30% of the backbone unfrozen with cosine LR | [section 7](README.md#7-training-protocol-and-curves) |
| Handling of class imbalance | focal loss; per-class thresholds on validation | [sections 8 and 10](README.md#10-threshold-optimisation) |
| Proper evaluation | test macro AUC 0.7012, micro AUC 0.7974, per-disease AUC with CIs, AP, F1, confusion counts | [sections 8-11](README.md#8-model-comparison) |
| Explainability | Grad-CAM on the same model | [section 12](README.md#12-explainability-with-grad-cam) |
| Error analysis, calibration, TTA | 1,248 FP / 354 FN breakdown, reliability diagrams, TTA on 200 images | [sections 11 and 13](README.md#11-confusion-matrices-and-error-analysis) |
| Runnable source code, tests, reproducibility | [`code/src`](../../code/README.md), unit tests, requirements, seeds, run guide | [`code/`](../../code/RUN_GUIDE.md) |
| Results stored as data | 19 CSV/JSON files | [`results/`](results/README.md) |

## CO4 - Compare with the state of the art (5 marks)

| Expected evidence | Delivered | Location |
|:---|:---|:---|
| Identify relevant published methods | Wang et al. 2017, Yao et al. 2017, Rajpurkar et al. 2017 (CheXNet) | [README section 15](README.md#15-comparison-with-published-methods-co4) |
| Quantitative comparison on the same benchmark | per-class and mean AUROC on all 14 classes and on the 12 well-populated classes | [`results/sota_comparison.csv`](results/sota_comparison.csv), [`results/sota_per_class_auroc.csv`](results/sota_per_class_auroc.csv) |
| Fair framing of differences | training-set size, split, resolution and architecture differences are stated explicitly | section 15 |
| Gap analysis | this project is 0.1402 mean AUROC below CheXNet (0.7012 vs 0.8414); closest class: Edema (-0.015); largest gaps: Emphysema (-0.268), Mass (-0.257) and Cardiomegaly (-0.249) | figures `fig_sota_per_class`, `fig_gap_to_chexnet` |
| Own competing baseline | four internal models plus ablation (DenseNet121 + ASL, ensemble) | sections 8 and 14 |

## CO5 - Innovation (5 marks)

| Expected evidence | Delivered | Location |
|:---|:---|:---|
| A novel or non-trivial improvement with justification | hypothesis-driven combination: **A** asymmetric loss, **B** deeper fine-tuning, **C** EfficientNetV2B0 + DenseNet121 probability ensemble; plus patient-level splitting, validation-only thresholds and Grad-CAM | [section 14](README.md#14-innovation-and-ablation-co5) |
| Controlled experiment | five runs with identical data, optimiser and schedule; one factor added at a time | [`results/ablation.csv`](results/ablation.csv) |
| Quantified benefit | 12-class macro AUC 0.7312 vs 0.6955 reference (+0.0132 over its best single member), macro F1 0.1843 vs 0.1361 | `fig_ablation`, `fig_ablation_val_auc` |
| Critical assessment | gains of A and B are within +-0.002 AUC; only the ensemble helps; single seed, double cost; deployable model remains the single network | section 14, "Honest assessment" |

> **Medical disclaimer.** Educational research project. The model is not a medical device and must not be used for diagnosis or treatment decisions.
