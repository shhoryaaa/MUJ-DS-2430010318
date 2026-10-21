# Assignment 4 — Transfer Learning and Fine-Tuning

## Transfer learning
A model trained on a large source dataset provides reusable visual features.
The notebook freezes EfficientNetV2B0 and trains a new classification head.

## Fine-tuning
Selected later layers are unfrozen and updated with a small learning rate.
This allows task-specific adaptation while preserving much of the pretrained
representation. Batch-normalization layers remain frozen in the notebook.

## Baseline comparison
The baseline CNN learns its representation from scratch. Comparing it with a
pretrained network helps assess whether transfer learning is useful for this
sample and training setup.

## Controls
Use the same partitions and label order. Select checkpoints using validation
metrics. Report training configuration and do not compare validation metrics
as though they were test scores.
