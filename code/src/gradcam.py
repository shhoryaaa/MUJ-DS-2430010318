"""Grad-CAM for the nested-backbone classifier (notebook cell 24)."""
import numpy as np


def make_gradcam(model, class_index, last_conv="top_activation", backbone_name="efficientnetv2-b0"):
    """Return f(img_batch)->(heatmap 7x7 in [0,1], probabilities). Backbone and head are called separately."""
    import tensorflow as tf
    base = model.get_layer(backbone_name)
    conv_extractor = tf.keras.Model(base.input, base.get_layer(last_conv).output)
    head = [model.get_layer(n) for n in ("gap", "bn_head", "dropout_head", "dense_head")]

    def run(img_batch):
        x = tf.convert_to_tensor(img_batch, tf.float32)
        with tf.GradientTape() as tape:
            fmap = conv_extractor(x, training=False)
            tape.watch(fmap)
            h = fmap
            for layer in head:
                h = layer(h, training=False)
            score = h[:, class_index]
        grads = tape.gradient(score, fmap)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0].numpy()
        return cam / (cam.max() + 1e-8), h.numpy()[0]
    return run
