
"""Evaluate the single production model and detect label/preprocessing mismatches."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from PIL import Image

from app import predict as predictor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Map differently named test folders to the official model class names
CLASS_FOLDER_MAP = {
    "Cherry___Powdery_mildew": "Cherry_(including_sour)___Powdery_mildew",
    "Cherry___healthy": "Cherry_(including_sour)___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn___healthy": "Corn_(maize)___healthy",
}

# This folder is not one of the model's 38 PlantVillage classes
SKIP_FOLDERS = {"Background_without_leaves"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, required=True, help="Class-per-folder independent test directory")
    p.add_argument("--output", type=Path, default=Path("evaluation_predictions.csv"))
    args = p.parse_args()

    model = predictor.load_assets()
    if model is None:
        raise RuntimeError(predictor.MODEL_LOAD_ERROR)
    
    folders = sorted(
        p for p in args.test_dir.iterdir()
        if p.is_dir() and p.name not in SKIP_FOLDERS
    )

    # Convert test-folder names into the model's official class names
    mapped_names = {
        p.name: CLASS_FOLDER_MAP.get(p.name, p.name)
        for p in folders
    }

    unknown = sorted(
        mapped
        for mapped in mapped_names.values()
        if mapped not in predictor.CLASS_NAMES
    )

    if unknown:
        raise RuntimeError(
            f"Unknown test classes after mapping: {unknown}"
        )

    rows, correct = [], 0
    matrix = np.zeros((len(predictor.CLASS_NAMES), len(predictor.CLASS_NAMES)), dtype=int)
    name_to_idx = {n: i for i, n in enumerate(predictor.CLASS_NAMES)}
    for folder in folders:
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            true_name = mapped_names[folder.name]
            true_idx = name_to_idx[true_name]

            pred_name, confidence, _, _, top = predictor.predict(Image.open(path).convert("RGB"), top_k=3)
            pred_idx = name_to_idx[pred_name]
            matrix[true_idx, pred_idx] += 1
            correct += int(pred_idx == true_idx)
            rows.append({"path": str(path), "true_class": true_name, "predicted_class": pred_name,
                        "confidence": confidence, "top3": json.dumps(top)})

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    accuracy = correct / len(rows) if rows else 0.0
    print(f"Images: {len(rows)}")
    print(f"Accuracy: {accuracy:.4f}")
    for i, name in enumerate(predictor.CLASS_NAMES):
        total = matrix[i].sum()
        recall = matrix[i, i] / total if total else 0.0
        print(f"{name}: recall={recall:.4f} ({matrix[i,i]}/{total})")
        print(f"Predictions written to {args.output}")

if __name__ == "__main__":
    main()
