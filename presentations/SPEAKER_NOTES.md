# Speaker Notes

## 1. Title
Introduce the project as an academic multi-label chest X-ray classification
prototype using EfficientNetV2B0 and explainability.

## 2. Motivation
A chest radiograph can contain several findings. The task therefore predicts
independent disease labels rather than choosing exactly one class.

## 3. Dataset
Explain the NIH ChestX-ray14 sample and its 14 target findings. Clarify that the
project uses a sample and does not redistribute the image data.

## 4. Data exploration
Discuss label imbalance and why rare labels need per-class evaluation.

## 5. Preprocessing
Describe metadata matching, multi-hot labels, image resizing and training
augmentation.

## 6. Patient-level split
Explain how assigning patients, not images, to partitions reduces leakage risk.

## 7. Baseline CNN
Describe convolutional feature extraction and the 14-unit sigmoid output.

## 8. EfficientNetV2B0
Explain transfer learning from pretrained image features and a task-specific head.

## 9. Fine-tuning
Explain unfreezing later layers and using a smaller learning rate.

## 10. Class imbalance
Describe focal loss as an experiment whose effect must be measured.

## 11. Evaluation
Differentiate validation results from held-out test results. Mention AUC, precision,
recall and F1 and explain that thresholds matter.

## 12. Grad-CAM
Describe heatmaps as qualitative visual aids, not proof of medical reasoning.

## 13. Limitations
Mention sample size, noisy labels, rare classes and lack of external clinical validation.

## 14. Conclusion
Summarize the pipeline and explain what further testing would be needed.
