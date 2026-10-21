"""Utilities for converting NIH-style finding strings to multi-hot targets."""

from typing import Iterable, Sequence
import numpy as np

DEFAULT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

def parse_finding_string(value) -> set[str]:
    """Parse a pipe-separated label string into a set of labels."""
    if value is None:
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return set()
    return {item.strip() for item in text.split("|") if item.strip()}

def to_multihot(value, labels: Sequence[str] = DEFAULT_LABELS) -> np.ndarray:
    """Convert a finding string to a binary vector in the given label order."""
    present = parse_finding_string(value)
    return np.asarray([int(label in present) for label in labels], dtype=np.uint8)

def batch_to_multihot(values: Iterable, labels=DEFAULT_LABELS) -> np.ndarray:
    """Convert an iterable of label strings to an (n_samples, n_labels) array."""
    rows = [to_multihot(value, labels) for value in values]
    if not rows:
        return np.zeros((0, len(labels)), dtype=np.uint8)
    return np.stack(rows)

def validate_multihot(y, n_labels: int = 14) -> None:
    """Raise ValueError when a target matrix has invalid dimensions or values."""
    arr = np.asarray(y)
    if arr.ndim != 2 or arr.shape[1] != n_labels:
        raise ValueError(f"Expected shape (n_samples, {n_labels}); got {arr.shape}")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("Targets must contain only 0 and 1.")
