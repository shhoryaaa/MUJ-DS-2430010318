# Dataset Card — NIH ChestX-ray14 Sample

## Intended use
Academic experimentation with chest-radiograph multi-label classification.

## Source
NIH ChestX-ray14 dataset, using the sample variant downloaded through KaggleHub
in the project notebook. Locate the authorized dataset source and read its
current terms before use.

## Notebook-reported scope
- Matched images: 5,606.
- Unique patients: 4,230.
- Image dimensions sampled in the notebook: 1024 × 1024.
- Target findings: 14.
- Image partitions reported after the notebook's patient-level split and cap:
  train 3,931; validation 849; test 826.

Counts are specific to the exported run and must be rechecked after changes.

## Target labels
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
Pleural_Thickening and Hernia.

## Known issues
- Unequal label frequencies.
- Potential label noise or incomplete labels.
- Multiple images can belong to the same patient.
- A label does not provide a precise pixel-level lesion location.
- A sample may not represent the full dataset distribution.

## Processing
The notebook maps image filenames, converts findings into binary columns,
checks image readability, creates patient-level splits, resizes to 224 × 224
and applies modest augmentation to training images.

## Distribution and privacy
Do not commit the raw dataset to this repository. Respect source terms, privacy
requirements and institutional policies.
