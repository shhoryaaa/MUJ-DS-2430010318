"""Audit patient-level split isolation."""

def audit_patient_splits(train_patients, val_patients, test_patients):
    train, val, test = map(set, (train_patients, val_patients, test_patients))
    overlaps = {
        "train_validation": train & val,
        "train_test": train & test,
        "validation_test": val & test,
    }
    report = {name: sorted(values) for name, values in overlaps.items()}
    report["passed"] = all(not values for values in overlaps.values())
    return report

def assert_patient_isolation(train_patients, val_patients, test_patients):
    report = audit_patient_splits(train_patients, val_patients, test_patients)
    if not report["passed"]:
        details = {k: v[:10] for k, v in report.items()
                   if k != "passed" and v}
        raise AssertionError(f"Patient overlap detected: {details}")
    return report
