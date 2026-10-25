"""Metadata discovery, label parsing, patient-level split and the tf.data pipeline (notebook cells 3, 6-9)."""
import os
import numpy as np
import pandas as pd
from .config import LABELS, SEED, IMG_SIZE, BATCH_SIZE


def find_metadata_csv(root):
    """Return the CSV under `root` that has a 'Finding Labels'-like column (largest file wins)."""
    cands = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".csv"):
                p = os.path.join(dirpath, f)
                try:
                    cols = pd.read_csv(p, nrows=1).columns
                except Exception:
                    continue
                if any("finding" in c.lower() for c in cols):
                    cands.append(p)
    if not cands:
        raise FileNotFoundError("no metadata CSV with a 'Finding Labels' column under " + root)
    return sorted(cands, key=lambda p: -os.path.getsize(p))[0]


def index_images(root):
    """Map image file name -> full path for every png/jpg below `root`."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                out[f] = os.path.join(dirpath, f)
    return out


def load_metadata(root):
    """Load the CSV, attach image paths, drop duplicates/missing files, parse the 14 binary label columns."""
    df = pd.read_csv(find_metadata_csv(root))
    img_col = next(c for c in df.columns if "image" in c.lower() and "index" in c.lower())
    finding_col = next(c for c in df.columns if "finding" in c.lower() and "label" in c.lower())
    patient_col = next(c for c in df.columns if "patient" in c.lower() and "id" in c.lower())
    df["full_path"] = df[img_col].map(index_images(root))
    df = df.dropna(subset=["full_path"]).drop_duplicates(subset=[img_col]).reset_index(drop=True)
    for d in LABELS:
        df[d] = df[finding_col].astype(str).apply(lambda s, d=d: int(d in s.split("|")))
    return df, patient_col


def patient_split(df, patient_col, seed=SEED, train_frac=0.70, val_frac=0.15):
    """Split by PATIENT (not image). Returns train, val, test frames and asserts zero patient overlap."""
    patients = df[patient_col].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_tr, n_va = int(train_frac * n), int(val_frac * n)
    sets = (set(patients[:n_tr]), set(patients[n_tr:n_tr + n_va]), set(patients[n_tr + n_va:]))
    parts = [df[df[patient_col].isin(s)].reset_index(drop=True) for s in sets]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (sets[i] & sets[j]), "patient leakage between splits"
    return parts[0], parts[1], parts[2]


def make_dataset(df, augment=False, shuffle=False, batch_size=BATCH_SIZE):
    """tf.data pipeline: read -> decode (3 ch) -> resize 224 -> float32 [0,255] -> (augment) -> batch -> prefetch."""
    import tensorflow as tf
    rot = tf.keras.layers.RandomRotation(0.02, seed=SEED)
    trans = tf.keras.layers.RandomTranslation(0.05, 0.05, seed=SEED)
    zoom = tf.keras.layers.RandomZoom(0.05, seed=SEED)

    def augment_image(img):  # no horizontal flip on purpose (anatomical laterality)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.squeeze(zoom(trans(rot(tf.expand_dims(img, 0), training=True), training=True), training=True), 0)
        return tf.clip_by_value(img, 0.0, 255.0)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.cast(tf.image.resize(img, [IMG_SIZE, IMG_SIZE]), tf.float32)
        return (augment_image(img) if augment else img), label

    ds = tf.data.Dataset.from_tensor_slices((df["full_path"].values, df[LABELS].values.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
