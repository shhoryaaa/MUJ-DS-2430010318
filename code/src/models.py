"""Architectures and losses (notebook cells 10, 12, 15, 17 and the CO5 ablation cell)."""
from .config import IMG_SIZE, NUM_CLASSES


def build_baseline_cnn():
    """3 x [Conv-BN-ReLU-MaxPool-Dropout(0.2)] with 32/64/128 filters, GAP, Dropout(0.3), Dense(14, sigmoid); 95,950 parameters."""
    import tensorflow as tf
    L = tf.keras.layers
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = L.Rescaling(1.0 / 255.0)(inp)
    for f in (32, 64, 128):
        x = L.Dropout(0.2)(L.MaxPooling2D()(L.Activation("relu")(L.BatchNormalization()(L.Conv2D(f, 3, padding="same")(x)))))
    x = L.Dropout(0.3)(L.GlobalAveragePooling2D()(x))
    return tf.keras.Model(inp, L.Dense(NUM_CLASSES, activation="sigmoid", dtype="float32")(x), name="baseline_cnn")


def build_classifier(backbone="effnetv2b0"):
    """ImageNet backbone (frozen) + GAP -> BatchNorm -> Dropout(0.3) -> Dense(14, sigmoid). Returns (model, base)."""
    import tensorflow as tf
    L = tf.keras.layers
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), name="input_image")
    if backbone == "effnetv2b0":
        base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = inp  # EfficientNetV2 rescales internally ([0,255] input)
    else:
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = L.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])(L.Rescaling(1 / 255.0)(inp))
    base.trainable = False
    x = base(x, training=False)
    x = L.Dropout(0.3, name="dropout_head")(L.BatchNormalization(name="bn_head")(L.GlobalAveragePooling2D(name="gap")(x)))
    out = L.Dense(NUM_CLASSES, activation="sigmoid", dtype="float32", name="dense_head")(x)
    return tf.keras.Model(inp, out, name=backbone + "_cxr"), base


def unfreeze_top(base, fraction=0.3):
    """Unfreeze the last `fraction` of backbone layers; BatchNorm layers stay frozen."""
    import tensorflow as tf
    cut = int(len(base.layers) * (1 - fraction))
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cut and not isinstance(layer, tf.keras.layers.BatchNormalization)


def binary_focal_loss(gamma=2.0, alpha=0.25):
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        bce = -(y_true * tf.math.log(p) + (1 - y_true) * tf.math.log(1 - p))
        p_t = y_true * p + (1 - y_true) * (1 - p)
        a = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(tf.reduce_sum(a * tf.pow(1.0 - p_t, gamma) * bce, axis=-1))
    return loss_fn


def asymmetric_loss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        pm = tf.clip_by_value(p - clip, 0.0, 1.0)  # probability shifting for negatives
        lp = y_true * tf.pow(1 - p, gamma_pos) * tf.math.log(p)
        ln = (1 - y_true) * tf.pow(pm, gamma_neg) * tf.math.log(tf.clip_by_value(1 - pm, 1e-7, 1.0))
        return -tf.reduce_mean(tf.reduce_sum(lp + ln, axis=-1))
    return loss_fn
