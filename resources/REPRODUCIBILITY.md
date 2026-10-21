# Reproducibility Guide

## Recorded notebook setup
The exported notebook reports Python 3.13.15, TensorFlow 2.21.0 and a Tesla T4
GPU. It enables mixed precision when supported and sets a random seed of 42.

These versions reflect the exported run and may not be compatible with every
TensorFlow/Keras release. Record your actual environment rather than forcing
these versions without testing.

## Before running
1. Start a fresh Colab runtime.
2. Configure dataset access through the official dataset source.
3. Run notebook cells in order.
4. Confirm the metadata and image paths were detected correctly.
5. Check target label order and patient-level partition isolation.
6. Save model checkpoints and training histories.
7. Run the final evaluation on the held-out test partition.
8. Export figures and metric tables.

## Recommended artifacts
- Environment versions.
- Dataset version and access date.
- Split manifest containing non-sensitive identifiers only.
- Training histories.
- Model checkpoint filenames and checksums.
- Per-label metric table and support.
- ROC and precision-recall curves.
- Confusion matrices.
- Grad-CAM examples with model, class and score metadata.

## Security
Never save Kaggle credentials, GitHub tokens or private patient data in this
repository or in notebook output cells.
