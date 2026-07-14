# predict.py
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

# ------------------------------
# PATHS
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "checkpoints" / "best_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

#MODEL_PATH = Path.cwd().parents[0] / "checkpoints" / "best_model.h5"
#CLASS_NAMES_PATH = Path.cwd().parents[0] / "class_names.json"

# ------------------------------
# LOAD MODEL
# ------------------------------
model = load_model(str(MODEL_PATH))

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# ------------------------------
# PREPROCESS (MATCH TRAINING)
# ------------------------------
def preprocess_image(pil_image: Image.Image, target_size=(224, 224)):
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.fromarray(pil_image)

    img = pil_image.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32")  # DO NOT scale
    return np.expand_dims(arr, axis=0)

# ------------------------------
# PREDICTION
# ------------------------------
def predict(pil_image: Image.Image):
    img_array = preprocess_image(pil_image)

    # ==============================
    # SANITY CHECK (CONSOLE PRINT)
    # ==============================
    #print("SANITY CHECK → Pixel min:", img_array.min())
    #print("SANITY CHECK → Pixel max:", img_array.max())
    #print("SANITY CHECK → Shape:", img_array.shape)
    # ==============================

    preds = model.predict(img_array)
    preds0 = preds[0]

    class_idx = int(np.argmax(preds0))
    predicted_class = CLASS_NAMES[class_idx]
    confidence = float(preds0[class_idx])

    # Optional: top-5 predictions
    print("Top-5 predictions:")
    for i in np.argsort(preds0)[-5:][::-1]:
        print(f"{CLASS_NAMES[i]}: {preds0[i]:.4f}")

    return predicted_class, confidence, preds0, img_array
