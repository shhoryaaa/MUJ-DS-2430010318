# Detailed Methodology

## Task definition
For an input image `x`, the model returns a vector `p` with 14 values in [0, 1].
Each output estimates a label-specific score. The ground-truth vector is multi-hot:
multiple positions can be 1 for the same image.

## Data pipeline
1. Load metadata.
2. identify the image filename, finding-label column and patient-ID column.
3. Map each metadata row to an existing image path.
4. Create one binary target column per finding.
5. Remove duplicate image identifiers and unreadable images.
6. Assign patient IDs to partitions before training.
7. Resize images to 224 × 224.
8. Use batches and prefetching for training throughput.

## Augmentation
The notebook uses small rotation, translation, zoom, brightness and contrast
changes. Augmentation should be visually inspected and kept medically plausible.
Horizontal flipping is intentionally omitted in the notebook because laterality
and anatomical positioning can matter.

## Baseline model
Three convolutional blocks use 32, 64 and 128 filters. Each block applies
convolution, batch normalization, ReLU, max pooling and dropout. Global average
pooling reduces the spatial representation before the 14-output sigmoid layer.

## Transfer learning
EfficientNetV2B0 uses pretrained ImageNet weights. The backbone is frozen while
a new output head is trained. The notebook compiles with Adam and binary
cross-entropy and tracks multi-label AUC.

## Fine-tuning
The last portion of the backbone is unfrozen, except batch-normalization layers.
A lower learning rate and cosine decay schedule are used to limit destructive
updates to pretrained features.

## Focal loss
The experiment applies a binary focal loss with gamma 2.0 and alpha 0.25.
This is an experimental objective, not a guarantee of superior rare-class results.

## Explainability
Grad-CAM is generated with respect to a selected class output. Record the target
class, prediction score, true label, model checkpoint and image identifier
alongside each explanation.

## Reproducibility controls
- Seed: 42 in the notebook.
- Input size: 224 × 224.
- Batch size: 32 in the exported run.
- Patient-level split with overlap assertions.
- Validation-based checkpoint selection.
- Separate held-out test evaluation.
- Record package versions and hardware.
