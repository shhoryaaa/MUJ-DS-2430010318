# Future Work Roadmap

## Priority 1 — trustworthy evaluation
- Re-run final evaluation using the selected saved checkpoint.
- Save per-label ROC-AUC, average precision, precision, recall, F1 and support.
- Select thresholds on validation data only.
- Add confidence intervals or bootstrap estimates where feasible.
- Preserve a strict untouched test set.

## Priority 2 — robustness
- Test external data from a different source or institution.
- Inspect subgroup performance where metadata permits and ethical approval allows.
- Investigate image markers, borders and acquisition artifacts as potential shortcuts.
- Evaluate calibration and distribution shift.

## Priority 3 — model comparison
- Compare EfficientNetV2B0 against alternative compact pretrained models.
- Test image resolution and augmentation settings systematically.
- Compare binary cross-entropy and focal loss under a fixed evaluation protocol.
- Track experiment configuration and checkpoint hashes.

## Priority 4 — explainability
- Generate Grad-CAM for true positives, false positives, true negatives and false negatives.
- Record the target class and score for every explanation.
- Compare explanations across models and image transformations.
- Seek expert review before making any clinical interpretation.

## Priority 5 — engineering
- Add configuration files and command-line arguments.
- Add unit tests for label parsing, metrics and patient splitting.
- Add environment lock files and CI checks.
- Provide a clear model card and dataset card.
