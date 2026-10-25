"""Inference: python -m src.predict --model final_best_model.keras --thresholds optimized_thresholds.json --image xray.png"""
import argparse, json
import numpy as np
from .config import LABELS, IMG_SIZE


def predict(model_path, thresholds_path, image_path):
    import tensorflow as tf
    from .models import binary_focal_loss
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={"loss_fn": binary_focal_loss()})
    thr = json.load(open(thresholds_path))
    img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
    img = tf.cast(tf.image.resize(img, [IMG_SIZE, IMG_SIZE]), tf.float32)[None]
    probs = model.predict(img, verbose=0)[0]
    found = [d for d, p in zip(LABELS, probs) if p >= thr[d]]
    return dict(probabilities={d: float(p) for d, p in zip(LABELS, probs)}, predicted=found or ["No Finding"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Educational chest X-ray classifier - NOT a medical device")
    ap.add_argument("--model", required=True); ap.add_argument("--thresholds", required=True); ap.add_argument("--image", required=True)
    a = ap.parse_args()
    r = predict(a.model, a.thresholds, a.image)
    for d, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"{d:20s} {p:.3f}")
    print("Predicted:", ", ".join(r["predicted"]))
