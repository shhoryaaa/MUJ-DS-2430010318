"""Framework-free evaluation utilities (numpy / scikit-learn only) - unit-tested in code/tests."""
import math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from .config import LABELS


def tune_thresholds(y_val, p_val, grid=None, default=0.5):
    """Per-class threshold maximising F1 on the VALIDATION split (notebook cell 20: grid 0.05..0.93 step 0.02)."""
    grid = np.arange(0.05, 0.95, 0.02) if grid is None else grid
    out = {}
    for i, d in enumerate(LABELS):
        if y_val[:, i].sum() == 0:
            out[d] = default
            continue
        f1s = [f1_score(y_val[:, i], (p_val[:, i] >= t).astype(int), zero_division=0) for t in grid]
        out[d] = float(grid[int(np.argmax(f1s))])
    return out


def per_class_report(y_true, p, thresholds):
    """Precision/recall/F1/ROC-AUC/AP and confusion counts per disease at the given thresholds."""
    rows = []
    for i, d in enumerate(LABELS):
        yt, yp = y_true[:, i], (p[:, i] >= thresholds[d]).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        has = yt.sum() > 0
        rows.append(dict(disease=d, threshold=thresholds[d], precision=precision_score(yt, yp, zero_division=0),
                         recall=recall_score(yt, yp, zero_division=0), f1=f1_score(yt, yp, zero_division=0),
                         roc_auc=roc_auc_score(yt, p[:, i]) if has else float("nan"),
                         avg_precision=average_precision_score(yt, p[:, i]) if has else float("nan"),
                         tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), positives=int(yt.sum())))
    return rows


def hanley_ci(auc, n_pos, n_neg):
    """Approximate 95% confidence interval of an AUC (Hanley & McNeil 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)


def macro_auc(y_true, p, min_pos=1):
    """Macro ROC-AUC over classes with at least `min_pos` positives (min_pos=10 reproduces the 12-class metric)."""
    vals = [roc_auc_score(y_true[:, i], p[:, i]) for i in range(y_true.shape[1])
            if y_true[:, i].sum() >= min_pos and y_true[:, i].sum() < len(y_true)]
    return float(np.mean(vals))
