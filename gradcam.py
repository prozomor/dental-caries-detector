# --- Grad-CAM: خريطة حرارية تُظهر أين نظر النموذج (المهمة B3) ---

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import tensorflow as tf
from train_model import MODEL_PATH, IMAGE_SIZE, preprocess_image


def make_gradcam(image_path, model=None):
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    # آخر طبقة خرجها رباعي الأبعاد = القاعدة المدرّبة (خرجه آخر خريطة التفافية 7x7)
    layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            layer_name = layer.name
            break
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output])
    arr = preprocess_image(image_path)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr)
        p = preds[0][0]
        # نشرح الفئة التي اختارها النموذج فعلاً (لا دائماً healthy)
        score = tf.where(p > 0.5, p, 1.0 - p)
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()      # ReLU ثم تحويل numpy بطريقة صحيحة
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = tf.image.resize(heatmap[..., None], (IMAGE_SIZE, IMAGE_SIZE))[..., 0].numpy()
    return heatmap


def save_gradcam(image_path, out_path="gradcam_overlay.png"):
    heatmap = make_gradcam(image_path)
    img = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    r = np.clip(heatmap * 2, 0, 1)
    g = np.clip(1 - np.abs(heatmap - 0.5) * 2, 0, 1)
    b = np.clip(2 - heatmap * 2, 0, 1)
    heat_rgb = np.stack([r, g, b], axis=-1)
    overlay = np.clip(0.55 * img_arr + 0.45 * heat_rgb, 0, 1)
    tf.keras.utils.save_img(out_path, (overlay * 255).astype("uint8"))
    return out_path


if __name__ == "__main__":
    print(save_gradcam("data/test/caries/127.jpg"))