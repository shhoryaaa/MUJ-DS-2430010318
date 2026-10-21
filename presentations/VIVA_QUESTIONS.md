# Viva Questions and Suggested Answers

**Q1. Why is this multi-label classification?**  
One image can contain several findings, so each disease is represented by an
independent binary target.

**Q2. Why use sigmoid rather than softmax?**  
Sigmoid produces a score for each independent label. Softmax would force scores
to compete and sum to one, which is inappropriate when multiple findings coexist.

**Q3. Why use binary cross-entropy?**  
It is a standard loss for independent binary targets.

**Q4. Why split by patient?**  
Images from the same patient can be correlated. Patient-level splitting reduces
the risk of leakage between train and test partitions.

**Q5. What is transfer learning?**  
Using features learned on a source dataset and adapting them to a new task.

**Q6. What is fine-tuning?**  
Unfreezing selected pretrained layers and updating them on the target data.

**Q7. Why use a lower learning rate during fine-tuning?**  
To adapt pretrained weights gradually and reduce the risk of damaging useful features.

**Q8. Why is class imbalance a problem?**  
Common labels can dominate learning, while rare labels have fewer positive examples.

**Q9. What does focal loss do?**  
It down-weights easy examples so training can focus relatively more on hard examples.

**Q10. What is Grad-CAM?**  
A gradient-based method that highlights image regions associated with a selected score.

**Q11. Does Grad-CAM prove the model is correct?**  
No. It is an approximate visualization and needs careful interpretation.

**Q12. Why report macro F1?**  
It gives each label equal weight, helping reveal poor performance on rare labels.

**Q13. Why not use only accuracy?**  
Accuracy can hide poor performance in imbalanced multi-label settings.

**Q14. Are validation AUC values test results?**  
No. Validation data supports model selection; test data is reserved for final evaluation.

**Q15. Can this model diagnose patients?**  
No. It is an academic prototype without clinical validation.

**Q16. What would you do next?**  
External validation, calibration, threshold analysis, per-label error review and expert assessment.
