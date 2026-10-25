# Resources

## Dataset
| Item | Detail |
|:---|:---|
| Name | NIH ChestX-ray14 (Wang et al., CVPR 2017) |
| Used | Kaggle sample `nih-chest-xrays/sample` (5,606 images, 4,230 patients) |
| Full dataset | `nih-chest-xrays/data` (112,120 images, 30,805 patients, about 42 GB) |
| Labels | Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia and "No Finding" |
| Access | Kaggle account; accept the dataset terms; `kagglehub` |

## Results mirrored from the experiments
[`results/`](results/README.md) holds the machine-readable test-set metrics (JSON/CSV).

## Reading list
1. Wang X., Peng Y., Lu L., Lu Z., Bagheri M., Summers R. M. *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases.* CVPR 2017.
2. Rajpurkar P. et al. *CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning.* arXiv:1711.05225, 2017.
3. Yao L., Poblenz E., Dagunts D., Covington B., Bernard D., Lyman K. *Learning to diagnose from scratch by exploiting dependencies among labels.* arXiv:1710.10501, 2017.
4. Tan M., Le Q. V. *EfficientNetV2: Smaller models and faster training.* ICML 2021 (arXiv:2104.00298).
5. Huang G., Liu Z., van der Maaten L., Weinberger K. Q. *Densely connected convolutional networks.* CVPR 2017.
6. Lin T.-Y., Goyal P., Girshick R., He K., Dollar P. *Focal loss for dense object detection.* ICCV 2017.
7. Ridnik T. et al. *Asymmetric loss for multi-label classification.* ICCV 2021.
8. Selvaraju R. R. et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV 2017.
9. Guo C., Pleiss G., Sun Y., Weinberger K. Q. *On calibration of modern neural networks.* ICML 2017.
10. Hanley J. A., McNeil B. J. *The meaning and use of the area under a receiver operating characteristic (ROC) curve.* Radiology 143(1), 1982.
11. NIH Clinical Center. *ChestX-ray14 / NIH Chest X-rays* (Kaggle datasets `nih-chest-xrays/data` and `nih-chest-xrays/sample`).
12. Abadi M. et al. *TensorFlow: Large-scale machine learning on heterogeneous systems*; Chollet F. et al. *Keras*.

## Glossary
| Term | Meaning in this project |
| :--- | :--- |
| Multi-label | one image can carry zero, one or several of the 14 findings; each finding is an independent yes/no output (sigmoid), not one softmax class |
| ROC-AUC | probability that a random positive image receives a higher score than a random negative one; 0.5 = chance, 1.0 = perfect; threshold independent |
| Macro vs micro | macro averages the 14 per-disease values equally (rare diseases count as much as common ones); micro pools all image-label decisions (dominated by common diseases) |
| Precision / recall | precision = TP/(TP+FP), recall (sensitivity) = TP/(TP+FN); specificity = TN/(TN+FP); F1 = harmonic mean of precision and recall |
| Average precision (PR-AUC) | area under the precision-recall curve; its random-guess value equals the disease prevalence, so the lift over prevalence is informative for rare findings |
| Subset accuracy | an image counts as correct only if all 14 labels are right; 'predict nothing' already scores 0.563 here, so it is reported but not used for ranking |
| Patient-level split | all images of one patient go to the same split, which prevents the model from memorising patients |
| Transfer learning / fine-tuning | start from ImageNet weights; first train only the new head (frozen backbone), then unfreeze the top backbone layers at a low learning rate |
| Focal loss / ASL | losses that down-weight easy negatives so the many negative labels do not drown the few positives |
| Grad-CAM | gradient-weighted class activation map highlighting image regions that increased the score of a chosen class |
| TTA | test-time augmentation: average predictions over slightly perturbed copies of an image |
| Calibration | whether a predicted probability of 0.3 really corresponds to roughly 30% positives |

## Useful links
* [Project README](../assignments/dl-project/README.md) - [CO mapping](../assignments/dl-project/CO_MAPPING.md) - [Capstone docs](../capstone/README.md)
* [Colab notebook](https://colab.research.google.com/drive/1jgplvn6oFYI93MF0KZumF6Mfa_XRWkIv)
