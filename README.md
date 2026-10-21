# Multi-Label Chest X-Ray Disease Classification Using EfficientNetV2, Transfer Learning, Fine-Tuning and Explainable AI

**Student Name:** SHHORYA AGARWAL
**Registration Number:** 2430010318
**Section:** D

## 1. Problem Statement
Classify chest X-ray images into one or more of 14 possible thoracic
disease findings (multi-label classification, since a single X-ray can
show multiple conditions simultaneously, or none at all).

## 2. Objective
Build a transfer-learning pipeline on top of EfficientNetV2B0 to detect
multiple thoracic diseases from chest X-rays, address the severe class
imbalance inherent to this dataset, and explain the model's predictions
using Grad-CAM.

## 3. Dataset
**NIH ChestX-ray14 (sample subset)**, accessed via Kaggle: `nih-chest-xrays/sample`
(falls back to the full `nih-chest-xrays/data` if the sample is unavailable).

- Total images used in this run: 5606
- Unique patients: 4230
- Image source directory auto-detected at runtime (no hard-coded paths)
- Disease labels (14): Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
- Each image can have zero, one, or multiple positive disease labels
  ("No Finding" means all 14 labels are 0)

## 4. Data Split
Patient-level splitting is used (not image-level), because the same
patient can have multiple X-rays. Splitting by image could let the same
patient appear in both train and test, letting the model partly
"memorize" a patient rather than learn generalizable disease features.

- Training images: 3931
- Validation images: 849
- Test images: 826
- No patient appears in more than one split (verified programmatically)

## 5. Preprocessing
- Images resized to 224 x 224
- Cast to float32 (EfficientNetV2 preprocessing expects [0, 255] floats,
  handled internally by the model's Rescaling layer)
- Built as an efficient `tf.data` pipeline with prefetching

## 6. Data Augmentation
Applied only during training, using only medically sensible
transformations:
- Small random rotation, translation, and zoom
- Random brightness and contrast adjustment (simulates acquisition variability)
- Horizontal flipping deliberately **avoided** — chest X-rays are not
  left-right symmetric in a clinically meaningless way, and flipping
  could teach the model incorrect anatomical priors.

## 7. Models Trained
1. **Baseline CNN** — a simple custom convolutional network trained from scratch.
2. **EfficientNetV2B0 (Transfer, frozen backbone)** — pretrained ImageNet
   backbone frozen, only a new classification head trained.
3. **EfficientNetV2B0 (Fine-tuned)** — the last portion of the backbone
   unfrozen and fine-tuned at a low learning rate with cosine decay.
4. **EfficientNetV2B0 (Fine-tuned + Focal Loss)** — retrained with a
   binary focal loss to address the severe class imbalance across the
   14 diseases (some diseases appear in under 1% of images).

## 8. Handling Class Imbalance
Standard binary cross-entropy tends to reward the model for simply
predicting "no disease" everywhere, since most labels are rare. A
**focal loss** (gamma=2.0, alpha=0.25) was used instead, which
down-weights easy/majority examples and focuses learning on
hard/minority examples — a robust alternative to naive oversampling,
which risks duplicating or distorting rare medical images.

## 9. Threshold Optimization
A fixed 0.5 probability threshold is rarely optimal for imbalanced
multi-label problems. Per-disease thresholds were instead selected by
maximizing F1 on the **validation set only**, then applied to the
untouched test set — avoiding any leakage from tuning on test data.

Optimized thresholds (selected on validation set):
  Atelectasis: 0.11
  Cardiomegaly: 0.09
  Effusion: 0.31
  Infiltration: 0.13
  Mass: 0.11
  Nodule: 0.09
  Pneumonia: 0.05
  Pneumothorax: 0.23
  Consolidation: 0.05
  Edema: 0.21
  Emphysema: 0.11
  Fibrosis: 0.05
  Pleural_Thickening: 0.07
  Hernia: 0.05

## 10. Explainability — Grad-CAM
Grad-CAM was used to visualize which regions of each X-ray the model
focused on when making its prediction, built directly from the actual
selected best model so the explanation matches the real decision
boundary rather than a stale/untrained reference model.

## 11. Model Comparison
                                      Model  Accuracy (subset-exact)  Macro Precision  Macro Recall  Macro F1  Micro F1  ROC-AUC (macro)  Training time (s)
                      Model 1: Baseline CNN                 0.562954         0.000000      0.000000  0.000000  0.000000         0.593370              667.5
      Model 2: EffNetV2B0 Transfer (frozen)                 0.537530         0.124813      0.021083  0.031710  0.097264         0.685206              744.4
             Model 3: EffNetV2B0 Fine-tuned                 0.533898         0.120536      0.022514  0.033917  0.096386         0.687098              759.4
Model 4: EffNetV2B0 Fine-tuned + Focal Loss                 0.549637         0.172065      0.027069  0.044203  0.084772         0.701152              497.1

**Best model selected for downstream analysis:** Model 4: EffNetV2B0 Fine-tuned + Focal Loss

Note: plain subset-exact accuracy can be misleading here, since with 14
mostly-rare independent labels, predicting "no disease" everywhere
still yields high accuracy. Macro F1, Micro F1, and macro ROC-AUC are
more trustworthy indicators of real performance on this task.

## 12. Per-Disease Performance (Final Model)
           Disease  Precision   Recall       F1  ROC-AUC  Avg Precision (PR-AUC)  Positive count (test)
             Edema   0.333333 0.052632 0.090909 0.872628                0.126514                     19
      Pneumothorax   0.750000 0.081081 0.146341 0.804131                0.273989                     37
            Hernia   0.000000 0.000000 0.000000 0.792727                0.005814                      1
          Effusion   0.325581 0.145833 0.201439 0.779595                0.274501                     96
     Consolidation   0.000000 0.000000 0.000000 0.734440                0.115727                     34
       Atelectasis   0.000000 0.000000 0.000000 0.729277                0.173601                     72
         Pneumonia   0.000000 0.000000 0.000000 0.688970                0.027488                      9
      Cardiomegaly   0.000000 0.000000 0.000000 0.675983                0.233571                     12
         Emphysema   0.000000 0.000000 0.000000 0.668610                0.061722                     20
Pleural_Thickening   0.333333 0.043478 0.076923 0.647572                0.076515                     23
          Fibrosis   0.000000 0.000000 0.000000 0.635647                0.029067                     12
            Nodule   0.000000 0.000000 0.000000 0.628790                0.099085                     46
              Mass   0.000000 0.000000 0.000000 0.611007                0.109341                     43
      Infiltration   0.666667 0.055944 0.103226 0.546745                0.251464                    143

## 13. Error Analysis
False positives and false negatives were logged for every disease and
every test image. High-confidence mistakes were visualized separately.
Qualitatively, false positives concentrate on diseases with overlapping
visual presentation (e.g. Infiltration, Consolidation, and Atelectasis
all show similar hazy opacities), while false negatives concentrate on
the rarest classes (e.g. Hernia, Fibrosis) where the model has seen
very few positive training examples.

## 14. Test-Time Augmentation (TTA)
Predictions were also evaluated using test-time augmentation (averaging
predictions across slightly perturbed copies of each test image) on a
subset of the test set, to check whether TTA improves robustness over a
single forward pass.

## 15. Limitations
- Trained on a ~5,600-image sample of NIH ChestX-ray14 rather than the
  full 112,120-image dataset, due to Colab's free-tier disk and runtime
  budget — results would likely improve with the full dataset.
- Several diseases (e.g. Hernia) have very few positive examples,
  making reliable evaluation for those classes difficult.
- NIH ChestX-ray14 labels were extracted from radiology reports via
  NLP, so some label noise is expected.
- This model is for academic/educational purposes only and is not a
  diagnostic tool.

## 16. Future Work
- Train on the full 112,120-image dataset for more reliable rare-class performance.
- Try larger EfficientNetV2 variants or ensembling multiple backbones.
- Explore label-smoothing or asymmetric loss variants for extreme imbalance.

## 17. Folder Structure
