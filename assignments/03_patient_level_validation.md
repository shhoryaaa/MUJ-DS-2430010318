# Assignment 3 — Patient-Level Validation

## Why split by patient?
Multiple radiographs from one patient can be strongly correlated. If images from
the same patient appear in both training and test partitions, the evaluation can
overestimate performance on unseen patients.

## Procedure
1. Obtain the unique patient identifiers.
2. Shuffle identifiers with a fixed seed.
3. Assign identifiers to train, validation and test sets.
4. Map each image to the partition of its patient.
5. Verify all pairwise intersections are empty.
6. Apply any sampling only after patient assignment.
7. Report both image counts and unique patient counts.

## Assertions
`train_patient_ids ∩ val_patient_ids = ∅`
`train_patient_ids ∩ test_patient_ids = ∅`
`val_patient_ids ∩ test_patient_ids = ∅`

## Caveat
A random patient split is not necessarily stratified for rare labels. Inspect
label support in every partition and document classes with low support.
