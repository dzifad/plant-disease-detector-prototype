# gradcam.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from PIL import Image
import matplotlib
#import matplotlib.cm as cm

def find_last_conv_layer(model):
    """Find the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

def apply_colormap_on_image(original_image_pil, heatmap, alpha=0.5, colormap_name="jet"):
    """
    original_image_pil: PIL.Image RGB
    heatmap: 2D numpy array in [0,1]
    returns: PIL.Image RGB the overlay
    """
    # convert heatmap to RGB using matplotlib colormap
   # colormap = cm.get_cmap(colormap_name)
    colormap = matplotlib.colormaps.get_cmap(colormap_name)
    heatmap_rgb = colormap(heatmap)[:, :, :3]  # HxWx3 in 0-1
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Resize heatmap to original image size
    heatmap_pil = heatmap_pil.resize(original_image_pil.size, resample=Image.BILINEAR)

    # Blend
    blended = Image.blend(original_image_pil.convert("RGB"), heatmap_pil, alpha=alpha)
    return blended

def generate_gradcam(model, img_array, class_index=None, original_image_pil: Image.Image = None, last_conv_layer_name: str = None):
    """
    model: tf.keras model
    img_array: preprocessed numpy array shape (1,H,W,3) float32
    class_index: int (optional) if not provided argmax of predictions will be used
    original_image_pil: PIL Image for overlay (optional). If not provided, a PIL of size 224x224 will be used.
    last_conv_layer_name: optional str to force a layer
    returns: PIL.Image (RGB) heatmap overlay (if original provided) or color heatmap sized 224x224
    """
    # ensure tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # find conv layer if needed
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    # build grad model
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)

        # ---- unwrap list outputs ----
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # predictions should now be (1, num_classes)
        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_channel = predictions[:, class_index]


    # gradients of the class wrt conv outputs
    grads = tape.gradient(class_channel, conv_outputs)  # shape (1, H, W, C)

    # take mean for each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (C,)

    conv_outputs = conv_outputs[0]  # shape (H, W, C)
    pooled_grads = pooled_grads.numpy()

    # weighted sum of feature maps
    conv_outputs_np = conv_outputs.numpy()
    heatmap = np.tensordot(conv_outputs_np, pooled_grads, axes=([2], [0]))  # shape (H, W)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)

    # resize heatmap to 224x224 (model input)
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_resized.resize((img_array.shape[2], img_array.shape[1]), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized).astype("float32") / 255.0

    # If original image provided, overlay on that (better UX)
    if original_image_pil is not None:
        overlay = apply_colormap_on_image(original_image_pil, heatmap_resized, alpha=0.5)
        return overlay

    # otherwise return color heatmap PIL sized as model input
    colormap = cm.get_cmap("jet")
    heatmap_rgb = colormap(heatmap_resized)[:, :, :3]
    heatmap_rgb = (heatmap_rgb * 255).astype("uint8")
    return Image.fromarray(heatmap_rgb)
