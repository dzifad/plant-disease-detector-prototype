"""Single source of truth for model loading, validation, preprocessing and prediction."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps

try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    _TF_IMPORT_ERROR = str(exc)
else:
    _TF_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "checkpoints" / "production_model.keras"
CONFIG_PATH = BASE_DIR / "checkpoints" / "model_config.json"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
DEFAULT_SIZE = (224, 224)

model = None
MODEL_AVAILABLE = False
MODEL_LOAD_ERROR = None
CONFIG = {}
CLASS_NAMES = []


def _load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_assets():
    global model, MODEL_AVAILABLE, MODEL_LOAD_ERROR, CONFIG, CLASS_NAMES
    if model is not None:
        return model
    if tf is None:
        MODEL_LOAD_ERROR = f"TensorFlow import failed: {_TF_IMPORT_ERROR}"
        return None
    try:
        CONFIG = _load_json(CONFIG_PATH, {}) or {}
        CLASS_NAMES = CONFIG.get("class_names") or _load_json(CLASS_NAMES_PATH, []) or []
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
        if not CLASS_NAMES:
            raise RuntimeError("No class names found in model_config.json or class_names.json.")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        output_count = int(model.output_shape[-1])
        if output_count != len(CLASS_NAMES):
            raise RuntimeError(
                f"Model/label mismatch: model has {output_count} outputs but metadata has {len(CLASS_NAMES)} classes."
            )
        MODEL_AVAILABLE = True
        MODEL_LOAD_ERROR = None
    except Exception as exc:
        model = None
        MODEL_AVAILABLE = False
        MODEL_LOAD_ERROR = str(exc)
    return model


def target_size():
    size = CONFIG.get("image_size", DEFAULT_SIZE)
    return tuple(int(x) for x in size)


def preprocess_image(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(target_size(), Image.Resampling.BILINEAR)
    # Training model embeds EfficientNet preprocessing; do not normalize twice.
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, 0)


def predict(image: Image.Image, top_k=3):
    model_obj = load_assets()
    if model_obj is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "Model unavailable")
    array = preprocess_image(image)
    scores = np.asarray(model_obj.predict(array, verbose=0)[0], dtype=np.float32)
    order = np.argsort(scores)[::-1][:max(1, min(top_k, len(scores)))]
    top = [{"class_name": CLASS_NAMES[int(i)], "confidence": float(scores[int(i)]), "index": int(i)} for i in order]
    best = top[0]
    return best["class_name"], best["confidence"], scores, array, top
