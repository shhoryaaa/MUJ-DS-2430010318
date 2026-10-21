# Assignment 1 — Problem Definition and Scope

## Problem
Build an academic prototype that maps one chest X-ray to 14 independent
finding scores. Multiple findings may coexist in one image.

## Inputs
- Chest radiograph.
- Metadata with image identifier, patient identifier and finding labels.

## Outputs
A 14-element vector of sigmoid scores, one per finding.

## Scope
Included: data preparation, patient-level split, CNN baseline, pretrained
EfficientNetV2B0, fine-tuning, imbalance experiment, quantitative evaluation
and Grad-CAM inspection.

Excluded: clinical deployment, patient-specific advice, autonomous diagnosis,
hospital integration and claims of clinical validity.

## Success criteria
- Reproducible data preparation.
- No patient overlap between splits.
- Valid multi-label targets.
- At least one baseline and one pretrained model.
- Metrics reported by class and overall.
- Results linked to actual model checkpoints.
- Limitations and responsible use explicitly documented.

## Risks
Label noise, class imbalance, compute constraints, dataset access changes,
shortcut learning, poor calibration and weak generalization.
