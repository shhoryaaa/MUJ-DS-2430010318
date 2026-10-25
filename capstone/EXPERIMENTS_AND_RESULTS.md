# Experiments and results

Every table is generated from the recorded run. Machine-readable files: [`../assignments/dl-project/results/`](../assignments/dl-project/results/README.md) and [`../resources/results/`](../resources/results/README.md).

## E1 - Four-model comparison (test set, threshold 0.5)
| Model | Subset acc. | Macro prec. | Macro recall | Macro F1 | Micro F1 | Macro ROC-AUC | Train time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1: Baseline CNN (scratch) | 0.5630 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5934 | 667.5 |
| Model 2: EfficientNetV2B0 transfer (frozen) | 0.5375 | 0.1248 | 0.0211 | 0.0317 | 0.0973 | 0.6852 | 744.4 |
| Model 3: EfficientNetV2B0 fine-tuned | 0.5339 | 0.1205 | 0.0225 | 0.0339 | 0.0964 | 0.6871 | 759.4 |
| Model 4: EfficientNetV2B0 fine-tuned + focal loss | 0.5496 | 0.1721 | 0.0271 | 0.0442 | 0.0848 | 0.7012 | 497.1 |

| Model | Best validation macro AUC | Best epoch | Test macro AUC | Gap (val - test) |
| :--- | ---: | ---: | ---: | ---: |
| Model 1: Baseline CNN (scratch) | 0.5837 | 5 | 0.5934 | -0.0096 |
| Model 2: EfficientNetV2B0 transfer (frozen) | 0.6973 | 6 | 0.6852 | +0.0121 |
| Model 3: EfficientNetV2B0 fine-tuned | 0.7073 | 5 | 0.6871 | +0.0202 |
| Model 4: EfficientNetV2B0 fine-tuned + focal loss | 0.7097 | 4 | 0.7012 | +0.0086 |

Result: ranking by macro AUC is Model 4 (0.7012) > Model 3 (0.6871) > Model 2 (0.6852) > Model 1 (0.5934). **Honest reading of the results.** The model is clearly better than chance (macro AUC 0.70) and transfer learning is the single biggest improvement over a scratch CNN, but it is far from clinical quality: at validation-tuned thresholds the macro precision is only 0.12 and the macro F1 only 0.15, and published models trained on the full 112,120-image dataset reach a mean AUROC of about 0.84. All numbers are measured on a single patient-level test split of 826 images and a single training seed.

## E2 - Training logs
### Baseline CNN
| Epoch | Train AUC | Val AUC | Train loss | Val loss |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 0.5075 | 0.5290 | 0.2495 | 0.3389 |
| 2 | 0.5189 | 0.5461 | 0.1899 | 0.2398 |
| 3 | 0.5415 | 0.5603 | 0.1867 | 0.1938 |
| 4 | 0.5484 | 0.5767 | 0.1846 | 0.1926 |
| 5 | 0.5422 | 0.5837 | 0.1836 | 0.1876 |
### EfficientNetV2B0, frozen
| Epoch | Train AUC | Val AUC | Train loss | Val loss |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 0.5423 | 0.6256 | 0.5073 | 0.1861 |
| 2 | 0.5869 | 0.6741 | 0.2016 | 0.1794 |
| 3 | 0.6323 | 0.6824 | 0.1878 | 0.1786 |
| 4 | 0.6600 | 0.6773 | 0.1827 | 0.1815 |
| 5 | 0.6849 | 0.6953 | 0.1790 | 0.1817 |
| 6 | 0.7235 | 0.6973 | 0.1727 | 0.1799 |
### EfficientNetV2B0, fine-tuned
| Epoch | Train AUC | Val AUC | Train loss | Val loss |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 0.7467 | 0.7005 | 0.1669 | 0.1799 |
| 2 | 0.7352 | 0.7007 | 0.1671 | 0.1791 |
| 3 | 0.7470 | 0.7058 | 0.1672 | 0.1791 |
| 4 | 0.7542 | 0.7047 | 0.1654 | 0.1788 |
| 5 | 0.7476 | 0.7073 | 0.1652 | 0.1790 |
| 6 | 0.7597 | 0.7025 | 0.1667 | 0.1790 |
### EfficientNetV2B0 + focal loss
| Epoch | Train AUC | Val AUC | Train loss | Val loss |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 0.7617 | 0.7063 | 0.3703 | 0.4146 |
| 2 | 0.7588 | 0.7046 | 0.3649 | 0.4031 |
| 3 | 0.7547 | 0.7063 | 0.3665 | 0.4050 |
| 4 | 0.7613 | 0.7097 | 0.3516 | 0.4071 |

## E3 - Per-disease evaluation of the final model
| Disease | ROC-AUC | 95% CI (approx.) | Test + | Avg. precision | Random AP | AP lift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Edema | 0.8726 | 0.770 - 0.975 | 19 | 0.1265 | 0.0230 | 5.5x |
| Pneumothorax | 0.8041 | 0.718 - 0.890 | 37 | 0.2740 | 0.0448 | 6.1x |
| Hernia | 0.7927 | 0.264 - 1.000 | 1 | 0.0058 | 0.0012 | 4.8x |
| Effusion | 0.7796 | 0.723 - 0.836 | 96 | 0.2745 | 0.1162 | 2.4x |
| Consolidation | 0.7344 | 0.637 - 0.832 | 34 | 0.1157 | 0.0412 | 2.8x |
| Atelectasis | 0.7293 | 0.661 - 0.798 | 72 | 0.1736 | 0.0872 | 2.0x |
| Pneumonia | 0.6890 | 0.495 - 0.883 | 9 | 0.0275 | 0.0109 | 2.5x |
| Cardiomegaly | 0.6760 | 0.507 - 0.845 | 12 | 0.2336 | 0.0145 | 16.1x |
| Emphysema | 0.6686 | 0.537 - 0.800 | 20 | 0.0617 | 0.0242 | 2.5x |
| Pleural_Thickening | 0.6476 | 0.524 - 0.771 | 23 | 0.0765 | 0.0278 | 2.7x |
| Fibrosis | 0.6356 | 0.465 - 0.806 | 12 | 0.0291 | 0.0145 | 2.0x |
| Nodule | 0.6288 | 0.540 - 0.717 | 46 | 0.0991 | 0.0557 | 1.8x |
| Mass | 0.6110 | 0.519 - 0.703 | 43 | 0.1093 | 0.0521 | 2.1x |
| Infiltration | 0.5467 | 0.494 - 0.600 | 143 | 0.2515 | 0.1731 | 1.5x |
| **macro mean** | 0.7012 | - | 40.5 | 0.1327 | - | - |

| Disease | Precision @0.5 | Recall @0.5 | F1 @0.5 | Avg. precision | Test + |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Edema | 0.3333 | 0.0526 | 0.0909 | 0.1265 | 19 |
| Pneumothorax | 0.7500 | 0.0811 | 0.1463 | 0.2740 | 37 |
| Hernia | 0.0000 | 0.0000 | 0.0000 | 0.0058 | 1 |
| Effusion | 0.3256 | 0.1458 | 0.2014 | 0.2745 | 96 |
| Consolidation | 0.0000 | 0.0000 | 0.0000 | 0.1157 | 34 |
| Atelectasis | 0.0000 | 0.0000 | 0.0000 | 0.1736 | 72 |
| Pneumonia | 0.0000 | 0.0000 | 0.0000 | 0.0275 | 9 |
| Cardiomegaly | 0.0000 | 0.0000 | 0.0000 | 0.2336 | 12 |
| Emphysema | 0.0000 | 0.0000 | 0.0000 | 0.0617 | 20 |
| Pleural_Thickening | 0.3333 | 0.0435 | 0.0769 | 0.0765 | 23 |
| Fibrosis | 0.0000 | 0.0000 | 0.0000 | 0.0291 | 12 |
| Nodule | 0.0000 | 0.0000 | 0.0000 | 0.0991 | 46 |
| Mass | 0.0000 | 0.0000 | 0.0000 | 0.1093 | 43 |
| Infiltration | 0.6667 | 0.0559 | 0.1032 | 0.2515 | 143 |
| **macro mean** | 0.1721 | 0.0271 | 0.0442 | 0.1327 | - |

## E4 - Threshold optimisation
| Disease | Threshold | Precision | Recall | F1 | Accuracy | Specificity | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edema | 0.21 | 0.1000 | 0.1579 | 0.1224 | 0.9479 | 0.9665 | 0.8726 |
| Pneumothorax | 0.23 | 0.2353 | 0.3243 | 0.2727 | 0.9225 | 0.9506 | 0.8041 |
| Hernia | 0.05 | 0.0000 | 0.0000 | 0.0000 | 0.9964 | 0.9976 | 0.7927 |
| Effusion | 0.31 | 0.2652 | 0.3646 | 0.3070 | 0.8087 | 0.8671 | 0.7796 |
| Consolidation | 0.05 | 0.0904 | 0.4706 | 0.1517 | 0.7833 | 0.7967 | 0.7344 |
| Atelectasis | 0.11 | 0.1583 | 0.3056 | 0.2085 | 0.7978 | 0.8448 | 0.7293 |
| Pneumonia | 0.05 | 0.0508 | 0.3333 | 0.0882 | 0.9249 | 0.9315 | 0.6890 |
| Cardiomegaly | 0.09 | 0.2727 | 0.2500 | 0.2609 | 0.9794 | 0.9902 | 0.6760 |
| Emphysema | 0.11 | 0.0909 | 0.0500 | 0.0645 | 0.9649 | 0.9876 | 0.6686 |
| Pleural_Thickening | 0.07 | 0.0459 | 0.2174 | 0.0758 | 0.8523 | 0.8705 | 0.6476 |
| Fibrosis | 0.05 | 0.0323 | 0.0833 | 0.0465 | 0.9504 | 0.9631 | 0.6356 |
| Nodule | 0.09 | 0.1121 | 0.2609 | 0.1569 | 0.8438 | 0.8782 | 0.6288 |
| Mass | 0.11 | 0.1000 | 0.1163 | 0.1075 | 0.8995 | 0.9425 | 0.6110 |
| Infiltration | 0.13 | 0.1721 | 0.6643 | 0.2734 | 0.3886 | 0.3309 | 0.5467 |
| **macro mean** | - | 0.1233 | 0.2570 | 0.1526 | 0.8615 | 0.8798 | 0.7012 |

## E5 - Confusion counts
| Disease | TP | FP | FN | TN | Test + |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 22 | 117 | 50 | 637 | 72 |
| Cardiomegaly | 3 | 8 | 9 | 806 | 12 |
| Effusion | 35 | 97 | 61 | 633 | 96 |
| Infiltration | 95 | 457 | 48 | 226 | 143 |
| Mass | 5 | 45 | 38 | 738 | 43 |
| Nodule | 12 | 95 | 34 | 685 | 46 |
| Pneumonia | 3 | 56 | 6 | 761 | 9 |
| Pneumothorax | 12 | 39 | 25 | 750 | 37 |
| Consolidation | 16 | 161 | 18 | 631 | 34 |
| Edema | 3 | 27 | 16 | 780 | 19 |
| Emphysema | 1 | 10 | 19 | 796 | 20 |
| Fibrosis | 1 | 30 | 11 | 784 | 12 |
| Pleural_Thickening | 5 | 104 | 18 | 699 | 23 |
| Hernia | 0 | 2 | 1 | 823 | 1 |
| **total** | 213 | 1248 | 354 | 9749 | 567 |

## E6 - Error analysis, TTA
Total false positives 1,248; false negatives 354; high-confidence mistakes 331. TTA on 200 test images: macro AUC 0.7139 -> 0.7136, macro F1 0.0676 -> 0.0683 (no benefit).

## E7 - Ablation (CO5)
| Experiment | A: ASL | B: deep FT | C: ensemble | Macro AUC (12 cls) | Macro F1 | Micro F1 | Trainable params | Total params | Min |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EffNetV2B0 + BCE, 30% unfrozen (reference) | - | - | - | 0.6955 | 0.1361 | 0.2272 | 3,088,894 | 5,942,366 | 16.6 |
| +A  asymmetric loss (ASL) | yes | - | - | 0.6972 | 0.1309 | 0.1804 | 3,088,894 | 5,942,366 | 20.5 |
| +A+B  deeper fine-tuning (60% unfrozen) | yes | yes | - | 0.6958 | 0.1596 | 0.2125 | 4,970,030 | 5,942,366 | 19.4 |
| DenseNet121 + ASL (60% unfrozen) | yes | yes | - | 0.7180 | 0.1810 | 0.2041 | 5,169,166 | 7,055,950 | 27.5 |
| +A+B+C  proposed: EffNetV2B0 + DenseNet121 ensemble | yes | yes | yes | 0.7312 | 0.1843 | 0.2201 | 10,139,196 | 12,998,316 | 46.9 |

## E8 - Published comparison (CO4)
| Method | Training data | Classes scored | Mean AUROC (14) | Mean AUROC (12) |
| :--- | ---: | ---: | ---: | ---: |
| Wang et al. 2017 (ChestX-ray8, ResNet-50) | full 112,120 images | 14 / 14 | 0.7381 | 0.7445 |
| Yao et al. 2017 (DenseNet + LSTM) | full 112,120 images | 14 / 14 | 0.8027 | 0.8009 |
| Rajpurkar et al. 2017 (CheXNet, DenseNet-121) | full 112,120 images | 14 / 14 | 0.8414 | 0.8412 |
| This project - Model 4 (single model, saved) | 5,606-image sample | 14 / 14 | 0.7012 | 0.6945 |
| This project - proposed ensemble (CO5 experiment) | 5,606-image sample | 12 (>=10 test +) | - | 0.7312 |

| Disease | Wang et al. 2017 | Yao et al. 2017 | Rajpurkar et al. 2017 | This project (Model 4) | Delta vs CheXNet |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.7160 | 0.7720 | 0.8094 | 0.7293 | -0.0801 |
| Cardiomegaly | 0.8070 | 0.9040 | 0.9248 | 0.6760 | -0.2488 |
| Effusion | 0.7840 | 0.8590 | 0.8638 | 0.7796 | -0.0842 |
| Infiltration | 0.6090 | 0.6950 | 0.7345 | 0.5467 | -0.1878 |
| Mass | 0.7060 | 0.7920 | 0.8676 | 0.6110 | -0.2566 |
| Nodule | 0.6710 | 0.7170 | 0.7802 | 0.6288 | -0.1514 |
| Pneumonia | 0.6330 | 0.7130 | 0.7680 | 0.6890 | -0.0790 |
| Pneumothorax | 0.8060 | 0.8410 | 0.8887 | 0.8041 | -0.0846 |
| Consolidation | 0.7080 | 0.7880 | 0.7901 | 0.7344 | -0.0557 |
| Edema | 0.8350 | 0.8820 | 0.8878 | 0.8726 | -0.0152 |
| Emphysema | 0.8150 | 0.8290 | 0.9371 | 0.6686 | -0.2685 |
| Fibrosis | 0.7690 | 0.7670 | 0.8047 | 0.6356 | -0.1691 |
| Pleural_Thickening | 0.7080 | 0.7650 | 0.8062 | 0.6476 | -0.1586 |
| Hernia | 0.7670 | 0.9140 | 0.9164 | 0.7927 | -0.1237 |
| **mean, all 14** | 0.7381 | 0.8027 | 0.8414 | 0.7012 | -0.1402 |
| **mean, 12 classes (>=10 test +)** | 0.7445 | 0.8009 | 0.8412 | 0.6945 | -0.1467 |

## Figures
<table>
<tr><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_training_curves.png" width="440"><br><sub>Train/validation AUC (top) and loss (bottom) for the four main models</sub></td><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_model_comparison.png" width="440"><br><sub>Comparison of the four main models on the test set</sub></td></tr>
<tr><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_per_disease_auc_ci.png" width="440"><br><sub>Per-disease ROC-AUC on the test set; error bars are approximate 95% confidence intervals</sub></td><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_ap_vs_prevalence.png" width="440"><br><sub>Average precision (PR-AUC) versus the random-guess baseline per disease</sub></td></tr>
<tr><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_thresholds_f1.png" width="440"><br><sub>Selected per-disease thresholds and their effect on test F1</sub></td><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_confusion_grid.png" width="440"><br><sub>Per-disease confusion matrices (counts) at the optimised thresholds</sub></td></tr>
<tr><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_ablation.png" width="440"><br><sub>Innovation ablation: effect of ASL, deeper fine-tuning and the two-backbone ensemble</sub></td><td align="center" width="50%"><img src="../assignments/dl-project/figures/fig_sota_per_class.png" width="440"><br><sub>Per-class AUROC of three published methods and of this project</sub></td></tr>
</table>
