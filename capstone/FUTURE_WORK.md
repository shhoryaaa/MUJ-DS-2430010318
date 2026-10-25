# Future work (prioritised)

| # | Experiment | Hypothesis | Success criterion | Effort |
|:--|:---|:---|:---|:---|
| 1 | Train on the full 112,120 images (patient-level, official NIH split) | the 20x data gap is the main cause of the 0.14 AUROC gap to CheXNet | mean AUROC above 0.78 | high (disk, GPU hours) |
| 2 | 5-fold patient-level cross-validation, 3 seeds | current differences of about 0.01 are within noise | report mean +- sd; keep only effects larger than 2 sd | medium |
| 3 | Input resolution 320 and 512 px | small nodules and masses need more pixels | Nodule and Mass AUC increase by at least 0.05 | medium |
| 4 | Label-dependency models (graph head, LSTM) and view position as input | co-occurrence and AP/PA shortcuts carry signal | Infiltration and Atelectasis AUC increase | medium |
| 5 | Temperature scaling and expected calibration error | outputs are over-confident for rare classes | ECE falls with unchanged AUC | low |
| 6 | Localisation audit of Grad-CAM against the NIH bounding boxes, plus randomisation sanity checks | maps are not purely edge or marker driven | pointing-game accuracy above a random-map baseline | medium |
| 7 | Subgroup analysis by sex, age, view | performance is uneven across groups | AUC gap between groups below 0.03 | low |
| 8 | External validation (CheXpert or MIMIC-CXR) | the model generalises beyond one hospital | AUC drop below 0.05 | high |
| 9 | Distillation of the ensemble into one network | keep ensemble accuracy at single-model cost | at most 0.005 AUC lost | medium |

None of these results exist yet; they are planned experiments, listed with the evidence that would confirm or refute each hypothesis.
