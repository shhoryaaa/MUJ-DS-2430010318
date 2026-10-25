"""Staged training driver: python -m src.train --data-dir <kaggle folder> --out-dir runs/exp1  (needs a GPU runtime)."""
import argparse, json, os
import numpy as np
from .config import LABELS, SEED
from .data import load_metadata, patient_split, make_dataset
from .models import build_baseline_cnn, build_classifier, unfreeze_top, binary_focal_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--out-dir", default="runs/exp1")
    ap.add_argument("--head-epochs", type=int, default=6); ap.add_argument("--ft-epochs", type=int, default=6); ap.add_argument("--focal-epochs", type=int, default=4)
    a = ap.parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(a.out_dir, exist_ok=True)
    df, pcol = load_metadata(a.data_dir)
    tr, va, te = patient_split(df, pcol)
    print("train/val/test images:", len(tr), len(va), len(te))
    tr_ds, va_ds = make_dataset(tr, augment=True, shuffle=True), make_dataset(va)
    auc = lambda: [tf.keras.metrics.AUC(name="auc", multi_label=True)]
    es = lambda: tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)
    model, base = build_classifier("effnetv2b0")                                   # stage 1: frozen backbone
    model.compile(tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=auc())
    model.fit(tr_ds, validation_data=va_ds, epochs=a.head_epochs, callbacks=[es(), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)])
    unfreeze_top(base, 0.3)                                                        # stage 2: fine-tune top 30%
    steps = int(np.ceil(len(tr) / 32)) * a.ft_epochs
    model.compile(tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(1e-5, steps)), loss="binary_crossentropy", metrics=auc())
    model.fit(tr_ds, validation_data=va_ds, epochs=a.ft_epochs, callbacks=[es()])
    model.compile(tf.keras.optimizers.Adam(1e-5), loss=binary_focal_loss(2.0, 0.25), metrics=auc())   # stage 3: focal loss
    model.fit(tr_ds, validation_data=va_ds, epochs=a.focal_epochs, callbacks=[es()])
    from .evaluate import tune_thresholds, per_class_report
    yv = va[LABELS].values.astype("float32"); pv = model.predict(va_ds, verbose=0)
    thr = tune_thresholds(yv, pv)
    te_ds = make_dataset(te); yt = te[LABELS].values.astype("float32"); pt = model.predict(te_ds, verbose=0)
    json.dump(thr, open(os.path.join(a.out_dir, "optimized_thresholds.json"), "w"), indent=2)
    json.dump(per_class_report(yt, pt, thr), open(os.path.join(a.out_dir, "test_report.json"), "w"), indent=2)
    model.save(os.path.join(a.out_dir, "final_best_model.keras"))
    print("saved to", a.out_dir)


if __name__ == "__main__":
    main()
