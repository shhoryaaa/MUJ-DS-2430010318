# Presentation

Deck(s) in this folder: [`Chest_XRay_Disease_Classification_Presentation  -  Repaired.pptx`](Chest_XRay_Disease_Classification_Presentation%20%20-%20%20Repaired.pptx)

## Slide outline (what the deck covers)
1. Title and student details; 2. Problem and motivation; 3. Dataset and imbalance; 4. Patient-level split; 5. Models; 6. Training strategy; 7. Results: model comparison (macro AUC 0.593 -> 0.685 -> 0.687 -> 0.701); 8. Per-disease results; 9. Threshold optimisation (macro F1 0.044 -> 0.153); 10. Grad-CAM; 11. Error analysis and calibration; 12. Innovation ablation (ensemble 0.731 vs 0.696); 13. Comparison with CheXNet / Yao / Wang; 14. Limitations and ethics; 15. Conclusion and future work.

## Demonstration script (5 minutes)
1. Open the [project README](../assignments/dl-project/README.md) and show the key-numbers table.
2. Show `fig_pipeline.png`, then `fig_model_comparison.png` (transfer learning gives the big jump).
3. Show `fig_per_disease_auc_ci.png` and explain why intervals matter for rare classes.
4. Show the Grad-CAM figure and say what it does and does not prove.
5. Show `fig_sota_per_class.png` and the honest gap to CheXNet; end with limitations.
