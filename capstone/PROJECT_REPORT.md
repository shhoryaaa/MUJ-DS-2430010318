# Project Report

## Title
Multi-Label Chest X-Ray Disease Classification Using EfficientNetV2,
Transfer Learning, Fine-Tuning and Explainable AI

## 1. Abstract
This project explores automated multi-label classification of chest radiographs.
Unlike single-label classification, multi-label learning allows one radiograph
to be associated with multiple findings. The experimental pipeline uses a
sample of NIH ChestX-ray14, creates 14 binary disease targets, and compares a
baseline convolutional neural network with EfficientNetV2B0. Transfer learning
starts with a pretrained feature extractor whose weights are frozen; fine-tuning
then allows selected later layers to adapt to radiographic data. An imbalance-aware
focal-loss experiment and Grad-CAM visualizations provide additional analysis.
Evaluation should include threshold-aware classification metrics and ranking
metrics. This prototype is intended for academic investigation and is not a
clinical decision-making tool.

## 2. Problem statement
Chest radiographs can contain several abnormalities simultaneously. A model
designed to predict only one class cannot naturally represent this situation.
The task is to estimate an independent probability for each of 14 target findings
from one image. The dataset is also imbalanced: some findings occur much less
frequently than others. These properties make data splitting, evaluation choice,
threshold selection and interpretation important.

## 3. Objectives
- Build a repeatable image classification pipeline.
- Convert textual findings into multi-hot target vectors.
- Avoid patient overlap between training, validation and test sets.
- Establish a baseline CNN for comparison.
- Apply ImageNet-pretrained EfficientNetV2B0 using transfer learning.
- Fine-tune selected later layers using a smaller learning rate.
- Explore focal loss for class imbalance.
- Evaluate each disease and aggregate performance.
- Generate visual explanations for selected predictions.
- Document failure modes and responsible-use constraints.

## 4. Dataset and target representation
The notebook export reports 5,606 matched images from the NIH ChestX-ray14
sample and 4,230 unique patients. It lists 14 findings. A single image can have
more than one positive target. `No Finding` corresponds to all disease targets
being zero. These figures describe the exported notebook run and should be
rechecked if a different dataset version or filtering rule is used.

## 5. Data preparation
Metadata is inspected for image identifiers, patient identifiers and finding
labels. Each target is created by checking whether the corresponding finding
appears in the label string. Duplicate image records are removed, and image
readability is verified. Paths that cannot be matched to image files are excluded.
Image resizing and batching are handled by TensorFlow's data pipeline.

## 6. Patient-level splitting
Images belonging to the same patient are correlated. If images from one patient
appear in both training and test sets, performance may be overly optimistic.
The notebook assigns patient IDs to separate partitions and checks intersections.
The exported run reports 3,931 training, 849 validation and 826 test images.
These counts are post-sampling image counts, not patient counts. Re-run the
split audit if preprocessing or sampling is changed.

## 7. Model A — baseline CNN
The baseline uses convolutional blocks with 32, 64 and 128 filters, batch
normalization, ReLU activations, max pooling and dropout. Global average pooling
feeds a 14-unit sigmoid output. Binary cross-entropy is used for multi-label
targets. The baseline provides a reference for measuring the benefit of
pretrained representations.

## 8. Model B — EfficientNetV2B0 transfer learning
EfficientNetV2B0 is initialized with ImageNet pretrained weights and its backbone
is frozen. A task-specific head applies global average pooling, batch normalization,
dropout and a 14-unit sigmoid output. Initially, only the new head is trained.
This reduces the number of trainable parameters during the first stage.

## 9. Model C — fine-tuning
The notebook unfreezes selected later backbone layers and keeps batch-normalization
layers frozen. A lower learning rate is used so pretrained features adapt gradually.
Early stopping and model checkpoints help preserve a strong validation checkpoint.
Fine-tuning may improve performance, but it can also overfit, so held-out evaluation
is required.

## 10. Class imbalance
The disease labels have unequal frequencies. The notebook examines positive
frequencies and implements binary focal loss as an additional experiment.
Focal loss down-weights easy examples relative to harder examples. It is not
guaranteed to improve every disease class; it must be compared using per-class
metrics and a fixed evaluation protocol.

## 11. Evaluation
Report the following where the corresponding predictions and labels are available:
- ROC-AUC per label and macro average.
- Average precision / PR-AUC per label where supported.
- Precision, recall and F1 at a documented threshold.
- Micro and macro aggregation.
- Confusion matrices for each label, interpreted as one-vs-rest.
- Calibration or threshold analysis where feasible.

A threshold of 0.5 is a convenient baseline, not necessarily the best threshold
for every disease. Thresholds must be selected on validation data, not optimized
on the held-out test set. AUC is a ranking metric and does not itself provide a
clinical operating point.

## 12. Explainability
Grad-CAM produces a heatmap based on gradients for a selected output score.
The heatmap is overlaid on the corresponding image to inspect image regions
associated with a model output. It is a qualitative debugging aid and does not
prove that the model relies on medically valid evidence. Review explanations
for multiple correct and incorrect predictions and multiple classes.

## 13. Findings and interpretation
The exported notebook logs validation AUC values during training. Those logs
must not be mislabeled as test performance. Final conclusions should be based on
the saved evaluation output from the same model checkpoint and split. If final
test metrics are unavailable, explicitly state that the held-out test evaluation
remains to be completed.

## 14. Limitations
The experiment uses a sample rather than the complete dataset. Labels may be
noisy, the class distribution is imbalanced, and image-level annotations do not
necessarily localize disease. Results may not generalize across hospitals,
equipment, demographics or acquisition protocols. A pretrained network may
learn shortcuts. Grad-CAM is not a clinical validation method.

## 15. Conclusion
The project implements a multi-label classification workflow and compares a
baseline CNN with transfer learning and fine-tuning using EfficientNetV2B0.
It also explores focal loss and Grad-CAM. The contribution is an end-to-end
academic pipeline with patient-level splitting and evaluation documentation.
Clinical use would require substantially more validation, external testing,
calibration, governance and expert review.

## 16. Future work
- Evaluate on a larger and independent dataset.
- Add external-site testing and subgroup analysis.
- Compare per-label threshold strategies.
- Study calibration and decision-curve analysis.
- Compare additional pretrained architectures.
- Audit explanations and shortcut learning.
- Improve experiment tracking and checkpoint provenance.
