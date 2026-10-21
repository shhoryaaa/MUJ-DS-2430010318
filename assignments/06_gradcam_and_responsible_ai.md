# Assignment 6 — Grad-CAM and Responsible AI

## Purpose
Grad-CAM creates a class-specific heatmap from gradients flowing to a
convolutional feature map. It helps inspect image regions associated with a score.

## Suggested review
For each selected class, inspect true positives, false positives, false negatives
and examples with no positive target. Record prediction score and ground truth.

## Limitations
Heatmaps are approximate and depend on model layer choice and target score.
A visually convincing heatmap does not prove that the model has learned a
clinically meaningful relationship.

## Responsible use
Do not present a prototype score as a diagnosis. Do not upload sensitive images
or identifiable patient metadata to public services. External validation and
expert review are prerequisites for any clinical claims.
