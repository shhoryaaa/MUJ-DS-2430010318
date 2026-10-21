# Assignment 5 — Class Imbalance and Evaluation Metrics

## Imbalance
Some labels are common and others are rare. A model can perform poorly on a rare
class even when its aggregate score appears reasonable.

## Metrics
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- ROC-AUC evaluates ranking across thresholds.
- Average precision summarizes the precision-recall curve.

## Macro vs micro
Macro metrics average per-label results and give each label equal weight.
Micro metrics aggregate decisions across labels and emphasize common classes.

## Thresholds
Sigmoid outputs are probabilities/scores. A 0.5 threshold is a baseline only.
Choose any tuned threshold using validation data and report it clearly.

## Focal loss
Focal loss changes the contribution of easy and hard examples. Evaluate whether
it improves rare-class performance rather than assuming it will.
