# Limitations, Ethics and Responsible Use

## Dataset and labels
The project uses a sample of NIH ChestX-ray14. The findings are dataset labels,
not independent diagnoses made for this project. Label noise and incomplete
annotations can affect training and evaluation.

## Class imbalance
Rare labels can have very few positive test cases. AUC and F1 estimates may be
unstable when positive support is low. Report support alongside each metric.

## Generalization
A sample from one dataset does not demonstrate performance at a different hospital.
Equipment, patient populations, acquisition settings and reporting practices can
shift. External validation is necessary before clinical claims.

## Data leakage
Patient-level separation is important. Any future preprocessing, deduplication,
threshold selection or model selection must preserve the test set as a final
unseen evaluation partition.

## Explainability
Grad-CAM is an approximate visualization. A heatmap can look plausible without
faithfully explaining all model behavior. Use it for inspection, not as proof
of correctness or causality.

## Clinical safety
This project is a learning and research prototype. It must not be used to diagnose,
triage or treat patients. A qualified clinician must not rely on these predictions
without validated clinical systems and appropriate oversight.

## Privacy and repository hygiene
Do not commit patient-identifiable information, raw medical images unless the
dataset license and institutional policy explicitly permit redistribution, tokens,
credentials, or private data. Keep large datasets and model weights outside Git
unless their size and distribution terms are suitable.
