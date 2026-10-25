# Limitations and ethics

## Data and labels
* **Size.** 5,606 of 112,120 images; 7 diseases have fewer than 25 test positives, Hernia exactly one. Their metrics are statistically fragile (see the confidence intervals).
* **Weak labels.** Labels come from text mining of radiology reports; they are not pixel-verified. "Infiltration" is an especially vague term.
* **Single split, single seed.** No cross-validation; no between-run variance.
* **Thresholds on small validation sets** (for example 2 Hernia positives) are noisy.
* **Not comparable one-to-one with the literature** (different training set, split and resolution).

## Fairness and population
The sample contains 3,134 men and 2,472 women (55.9% / 44.1%) and a hospital population from one institution. Performance by sex, age and view position (AP versus PA) was not analysed; AP films are typical for sicker, bedridden patients and may act as a shortcut.

## Privacy and data governance
The NIH images are publicly released and de-identified by the NIH Clinical Center. This repository contains no raw images, only aggregated results and figures derived from them. The terms of the dataset on Kaggle must be respected when re-using it.

## Safe use
* The model is **not a medical device**, has not been clinically validated and must not be used to diagnose or exclude disease.
* At the tuned thresholds the macro precision is 0.12: most alarms are false. A clinical setting would need operating points chosen with clinicians.
* Probabilities are not calibrated risks. Grad-CAM maps are exploratory and do not prove clinical reasoning.
* Automation bias: users may over-trust a confident output; any deployment would require human review, monitoring and prospective evaluation.

## Honest reporting statement
All numbers in this repository were produced by one recorded run and are reproduced as printed; the earlier README of this repository contained inflated figures and has been corrected in the final update. Derived values (confusion counts, confidence intervals, lifts, SOTA means) are computed in code from the recorded numbers, with integrity checks against totals logged by the notebook.

> **Medical disclaimer.** Educational research project. The model is not a medical device and must not be used for diagnosis or treatment decisions.
