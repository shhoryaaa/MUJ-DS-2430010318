# Running the Helpers

## Environment
Use Google Colab or a Python environment with TensorFlow installed.
The full training notebook handles dataset acquisition and model training.
These scripts are helpers; they are not a replacement for the notebook.

## Label parsing
```python
from label_utils import to_multihot
y = to_multihot("Effusion|Mass")
print(y.shape)  # (14,)
```

## Patient split audit
```python
from split_audit import assert_patient_isolation
assert_patient_isolation(train_patient_ids, val_patient_ids, test_patient_ids)
```

## Metric calculation
```python
from metrics import calculate_metrics
result = calculate_metrics(y_test, y_prob, threshold=0.5, label_names=LABELS)
print(result["macro_f1"])
```

`y_prob` must be model probabilities for exactly the same ordered labels as
`y_test`. Choose tuned thresholds using validation data only.

## Training curves
```python
from plot_results import plot_history
plot_history(ft_history, "outputs/finetune_history.png", "EfficientNet fine-tuning")
```

The exact variable names in your active notebook may differ. Do not claim these
examples have been executed unless you run them successfully.
