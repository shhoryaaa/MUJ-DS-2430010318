# Experiments and Results

## Experiment matrix

| ID | Model | Initialization | Trainable component | Objective |
|---|---|---|---|---|
| E1 | Baseline CNN | Random | All CNN layers | Binary cross-entropy |
| E2 | EfficientNetV2B0 | ImageNet | New classification head | Binary cross-entropy |
| E3 | EfficientNetV2B0 fine-tuned | E2 checkpoint | Selected later backbone layers + head | Binary cross-entropy |
| E4 | Imbalance-aware model | Fine-tuned checkpoint | Matching trainable configuration | Binary focal loss |

## Notebook-reported validation history
The exported notebook logs the following maximum validation AUC values:
- Baseline CNN: 0.5837 at epoch 5.
- Frozen-backbone transfer learning: 0.6973 at epoch 6.
- Fine-tuning: 0.7073 at epoch 5.

These are values visible in the notebook's training logs. They are validation
metrics, not held-out test scores. They should not be compared with results from
different splits or data processing configurations.

## Final results table
Fill this table from the actual final test evaluation, not from training history.

| Model | Test macro ROC-AUC | Test micro ROC-AUC | Macro F1 | Micro F1 | Notes |
|---|---:|---:|---:|---:|---|
| Baseline CNN | Not recorded here | Not recorded here | Not recorded here | Not recorded here | Run evaluation script |
| EfficientNetV2B0 transfer learning | Not recorded here | Not recorded here | Not recorded here | Not recorded here | Run evaluation script |
| EfficientNetV2B0 fine-tuned | Not recorded here | Not recorded here | Not recorded here | Not recorded here | Run evaluation script |
| Focal-loss experiment | Not recorded here | Not recorded here | Not recorded here | Not recorded here | Run evaluation script |

## Required reporting details
Record the model checkpoint path, dataset version, split seed, number of test
images, label order, probability threshold, and software versions. Include
per-label results because aggregate scores can hide poor performance on rare
findings. Report uncertainty where feasible.

## Interpretation template
1. Did pretrained features improve validation ranking performance over baseline?
2. Did fine-tuning improve validation performance consistently or only at one epoch?
3. Did focal loss improve rare-label recall without unacceptable precision loss?
4. Which labels have the weakest support and most uncertain metrics?
5. Are any explanations concentrated on borders, text markers or acquisition artifacts?
6. Does the held-out test result support the same conclusion as validation?
