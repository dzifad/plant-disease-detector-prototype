"""Grad-CAM implementation for a classifier containing a nested CNN backbone."""

import numpy as np
from PIL import Image
import matplotlib

try:
    import tensorflow as tf
except Exception:
    tf = None


def build_connected_gradcam_model(model):
    """
    Rebuild the model graph using the existing trained layers and weights.

    This creates a graph in which the convolutional feature maps and final
    predictions are both connected to the same input. It resolves the
    'Output with path 0 is not connected to inputs' error caused by nested
    EfficientNet models.
    """

    if tf is None:
        raise RuntimeError("TensorFlow is not available.")

    if model is None:
        raise RuntimeError("The trained model is not loaded.")

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    new_input = tf.keras.Input(
        shape=input_shape[1:],
        name="gradcam_input",
    )

    x = new_input
    feature_maps = None
    feature_layer_name = None

    # Skip the original InputLayer and reconnect every other layer.
    for layer in model.layers[1:]:
        x = layer(x, training=False)

        # EfficientNet is normally stored as a nested Keras model.
        if isinstance(layer, tf.keras.Model) and len(x.shape) == 4:
            feature_maps = x
            feature_layer_name = layer.name

        # Fallback for a top-level convolutional feature layer.
        elif len(x.shape) == 4:
            feature_maps = x
            feature_layer_name = layer.name

    if feature_maps is None:
        raise ValueError(
            "No connected four-dimensional convolutional feature map "
            "was found in the model."
        )

    grad_model = tf.keras.Model(
        inputs=new_input,
        outputs=[feature_maps, x],
        name="connected_gradcam_model",
    )

    return grad_model, feature_layer_name


def apply_colormap_on_image(
    original_image,
    heatmap,
    alpha=0.45,
    colormap_name="jet",
):
    """
    Overlay a Grad-CAM heatmap on the original image.
    """

    colormap = matplotlib.colormaps.get_cmap(colormap_name)

    heatmap_rgb = colormap(heatmap)[:, :, :3]
    heatmap_rgb = np.uint8(heatmap_rgb * 255)

    heatmap_image = Image.fromarray(heatmap_rgb)
    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )

    return Image.blend(
        original_image.convert("RGB"),
        heatmap_image.convert("RGB"),
        alpha=alpha,
    )


def generate_gradcam(
    model,
    img_array,
    class_index=None,
    original_image_pil=None,
    last_conv_layer_name=None,
):
    """
    Generate a Grad-CAM heatmap for the requested prediction.

    Parameters
    ----------
    model:
        Loaded TensorFlow/Keras classification model.

    img_array:
        Input image array with shape (1, height, width, 3).

    class_index:
        Output class index. When omitted, the highest-scoring class is used.

    original_image_pil:
        Original PIL image used for the heatmap overlay.

    last_conv_layer_name:
        Retained for compatibility. The connected backbone output is used
        automatically.
    """

    if tf is None:
        raise RuntimeError("TensorFlow is not available.")

    if model is None:
        raise RuntimeError("The trained model is not loaded.")

    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32,
    )

    grad_model, feature_layer_name = build_connected_gradcam_model(model)

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(
            img_tensor,
            training=False,
        )

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_index = tf.cast(class_index, tf.int32)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            f"Gradients could not be calculated from feature layer "
            f"'{feature_layer_name}'."
        )

    # Average each feature-map gradient over height and width.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = feature_maps[0]

    # Weight feature channels by their importance to the selected class.
    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1,
    )

    # Retain only positive class contributions.
    heatmap = tf.nn.relu(heatmap)

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        tf.zeros_like(heatmap),
    )

    heatmap = heatmap.numpy()

    if original_image_pil is not None:
        return apply_colormap_on_image(
            original_image_pil,
            heatmap,
            alpha=0.45,
        )

    colormap = matplotlib.colormaps.get_cmap("jet")
    heatmap_rgb = colormap(heatmap)[:, :, :3]
    heatmap_rgb = np.uint8(heatmap_rgb * 255)

    return Image.fromarray(heatmap_rgb)