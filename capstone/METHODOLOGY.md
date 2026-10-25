# Methodology

## 1. Data acquisition and validation
The data are downloaded with `kagglehub.dataset_download("nih-chest-xrays/sample")`. The folder is scanned recursively: the metadata CSV is the file with a column containing "finding" (largest file wins); every `.png/.jpg` is indexed by file name and joined to the `Image Index` column. Rows without an image are dropped (0), duplicate rows are dropped (0), each image is opened once to detect corruption (0 failures).

## 2. Label encoding
`Finding Labels` such as `Cardiomegaly|Emphysema` is split on `|` and mapped to 14 binary columns; "No Finding" is the all-zero vector. Output order: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.

## 3. Patient-level split
```
patients <- unique(Patient ID); shuffle with RandomState(42)
train <- first 70% of patients; validation <- next 15%; test <- last 15%
every image inherits the split of its patient
assert train, validation and test patient sets are pairwise disjoint
```
Result: train 3,931 images / 2,961 patients, validation 849 / 634, test 826 / 635.

## 4. Pre-processing and augmentation
Read -> decode to 3 channels -> bilinear resize to 224 x 224 -> `float32` in [0, 255] -> (training only) random brightness (+-0.1), contrast (0.9-1.1), rotation (factor 0.02, about 7 degrees), translation (5%), zoom (5%) -> batch 32 -> prefetch. No horizontal flip because cardiac position and laterality-dependent findings make mirrored films anatomically implausible.

## 5. Models
| Model | Definition |
|:---|:---|
| Baseline CNN | 3 blocks of Conv(3x3, same) - BatchNorm - ReLU - MaxPool - Dropout 0.2 with 32/64/128 filters; GAP; Dropout 0.3; Dense 14 sigmoid. 95,950 parameters |
| EfficientNetV2B0 frozen | ImageNet backbone with `trainable=False` (called with `training=False`); GAP - BatchNorm - Dropout 0.3 - Dense 14 sigmoid (float32). 5,942,366 parameters |
| Fine-tuned | the last 30% of the backbone layers trainable, BatchNorm layers frozen, Adam with cosine decay from 1e-5 |
| Focal | fine-tuned weights, Adam 1e-5, focal loss |

## 6. Losses
Multi-label binary cross-entropy: $L = -\frac{1}{B}\sum_b \sum_{c=1}^{14} [y\log p + (1-y)\log(1-p)]$.

Focal loss (gamma = 2, alpha = 0.25): $L = \sum_c \alpha_t (1-p_t)^{\gamma} \cdot \mathrm{BCE}_c$ with $p_t = p$ for positives and $1-p$ for negatives, so well-classified labels contribute little.

Asymmetric loss (CO5): positives use $(1-p)^{\gamma_+}\log p$ with $\gamma_+ = 0$; negatives use the shifted probability $p_m = \max(p-0.05, 0)$ and $p_m^{\gamma_-}\log(1-p_m)$ with $\gamma_- = 4$.

## 7. Optimisation schedule
Stage 1: Adam 1e-3, 6 epochs, ReduceLROnPlateau(0.5, patience 2), EarlyStopping(val AUC, patience 3, restore best). Stage 2: 6 epochs, cosine decay from 1e-5. Stage 3: focal loss, 4 epochs. Baseline: Adam 1e-3, 5 epochs. Mixed precision (float16 compute) on a Tesla T4.

## 8. Threshold search
For every disease, scan t in {0.05, 0.07, ..., 0.93} on the validation split and keep the t with the largest F1; use 0.5 if the validation split had no positives (not the case). Thresholds are frozen before the test split is touched.

## 9. Metrics
Precision = TP/(TP+FP); recall = TP/(TP+FN); specificity = TN/(TN+FP); F1 = 2PR/(P+R); macro = unweighted mean over diseases; micro = pooled over image-label pairs. ROC-AUC via the rank statistic; average precision as area under the precision-recall curve. Subset accuracy = all 14 labels correct.

**Confidence intervals.** For an AUC $A$ with $n_1$ positives and $n_2$ negatives (Hanley and McNeil, 1982): $Q_1 = A/(2-A)$, $Q_2 = 2A^2/(1+A)$, $SE^2 = [A(1-A) + (n_1-1)(Q_1-A^2) + (n_2-1)(Q_2-A^2)]/(n_1 n_2)$, interval $A \pm 1.96\,SE$ (clipped to [0, 1]). It is an approximation and is optimistic for very small $n_1$.

**Confusion counts** at the tuned thresholds are recovered exactly from recorded recall and precision (TP = recall x positives; FP = TP/precision - TP; for TP = 0 from specificity); they sum to the 1,248 FP and 354 FN logged by the notebook.

## 10. Grad-CAM
Feature map $A^k$ of the last convolutional activation (`top_activation`, 7 x 7 x 1280); weights $\alpha_k = \frac{1}{Z}\sum_{i,j} \partial y^c / \partial A^k_{ij}$; map $= \mathrm{ReLU}(\sum_k \alpha_k A^k)$, normalised to [0, 1] and up-sampled onto the image. The gradient is taken through the same classification head (GAP, BatchNorm, Dropout in inference mode, Dense) as used for prediction.

## 11. TTA and calibration
TTA averages predictions over mildly augmented copies (200-image test subset). Reliability diagrams bin predictions and compare mean predicted probability with the observed frequency.

## 12. Ablation design (CO5)
Five runs with identical split, batch size, 3 head epochs (Adam 1e-3) and up to 6 fine-tuning epochs (Adam 1e-4): reference (BCE, 30% unfrozen); +A (ASL); +A+B (ASL, 60% unfrozen); DenseNet121 with the +A+B recipe; ensemble = mean of the +A+B EfficientNetV2B0 and DenseNet121 probabilities. Thresholds are tuned on validation for each; macro AUC is averaged over the 12 diseases with at least 10 test positives.
