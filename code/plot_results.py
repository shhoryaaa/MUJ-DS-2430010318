"""Plot training histories and multi-label metric summaries."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, output_path, title="Training history"):
    """Save loss and AUC curves from a Keras History object."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in hist:
        axes[0].plot(hist["loss"], label="train")
    if "val_loss" in hist:
        axes[0].plot(hist["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    auc_key = next((k for k in hist if "auc" in k and not k.startswith("val_")), None)
    if auc_key:
        axes[1].plot(hist[auc_key], label="train")
        if f"val_{auc_key}" in hist:
            axes[1].plot(hist[f"val_{auc_key}"], label="validation")
    axes[1].set_title("AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out)

def plot_per_label_scores(metrics, output_path, metric="f1"):
    """Save a horizontal bar chart of per-label metric values."""
    per_label = metrics["per_label"]
    names, values = [], []
    for name, item in per_label.items():
        value = item.get(metric)
        if value is not None:
            names.append(name)
            values.append(value)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    order = np.argsort(values)
    ax.barh(np.asarray(names)[order], np.asarray(values)[order])
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Per-label {metric.upper()}")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out)
