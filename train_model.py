"""
Train a plant disease classifier on the PlantVillage dataset.

Uses EfficientNetB0 transfer learning with a two-phase schedul
  1. Train the classification head with a frozen backbone.
  2. Fine-tune the top backbone layers at a lower learning rate.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

DATASET_ROOT = Path(r"C:\PlantData\PlantVillage-Dataset-master")

SVM_DIR = DATASET_ROOT / "data_distribution_for_SVM"
RAW_COLOR_DIR = DATASET_ROOT / "raw" / "color"

SAMPLES_DIR = BASE_DIR / "app" / "samples"

IMAGE_SIZE = (224, 224)
BACKBONE = "efficientnetb0"
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train the plant disease classifier.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Train on app/samples only (fast sanity check).",
    )
    parser.add_argument("--epochs-head", type=int, default=6, help="Epochs for head training.")
    parser.add_argument("--epochs-finetune", type=int, default=12, help="Epochs for fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    return parser.parse_args()


def parse_svm_mapping(mapping_path: Path) -> dict[str, str]:
    """Map numeric SVM folder ids to PlantVillage class folder names."""
    id_to_name: dict[str, str] = {}
    for line in mapping_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        raw_path, svm_path = line.split("\t", 1)
        raw_parts = raw_path.replace("\\", "/").split("/")
        svm_parts = svm_path.replace("\\", "/").split("/")
        if len(raw_parts) < 3 or len(svm_parts) < 3:
            continue
        id_to_name[svm_parts[2]] = raw_parts[2]
    return id_to_name


def collect_svm_split(split_dir: Path, id_to_name: dict[str, str]) -> tuple[list[str], list[int], list[str]]:
    """Collect image paths and integer labels from an SVM train/test split."""
    class_names = sorted(set(id_to_name.values()))
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    paths: list[str] = []
    labels: list[int] = []

    for folder in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if not folder.is_dir():
            continue
        class_name = id_to_name.get(folder.name)
        if class_name is None:
            continue
        label = name_to_idx[class_name]
        for image_path in folder.iterdir():
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(image_path))
                labels.append(label)

    return paths, labels, class_names


def collect_directory_split(data_dir: Path) -> tuple[list[str], list[int], list[str]]:
    """Collect images from a standard class-per-folder layout."""
    class_names = sorted(
        folder.name for folder in data_dir.iterdir() if folder.is_dir() and any(folder.iterdir())
    )
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    paths: list[str] = []
    labels: list[int] = []

    for class_name in class_names:
        label = name_to_idx[class_name]
        for image_path in (data_dir / class_name).iterdir():
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(image_path))
                labels.append(label)

    return paths, labels, class_names



def stratified_directory_split(
    data_dir: Path,
    validation_fraction: float = 0.20,
    seed: int = 42,
    minimum_images: int = 3,
) -> tuple[list[str], list[int], list[str], list[int], list[str]]:
    """Create a deterministic per-class train/validation split.

    Quick mode permits small sample folders. Every class contributes at least
    one image to validation and at least one image to training.
    """
    paths, labels, class_names = collect_directory_split(data_dir)
    rng = np.random.default_rng(seed)

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []

    paths_array = np.asarray(paths, dtype=object)
    labels_array = np.asarray(labels, dtype=np.int32)

    for class_index, class_name in enumerate(class_names):
        class_positions = np.flatnonzero(labels_array == class_index)
        if len(class_positions) < minimum_images:
            raise RuntimeError(
                f"Class {class_name!r} has only {len(class_positions)} images; "
                f"at least {minimum_images} are required for quick mode."
            )

        rng.shuffle(class_positions)
        validation_count = max(1, round(len(class_positions) * validation_fraction))
        validation_count = min(validation_count, len(class_positions) - 1)

        class_val_positions = class_positions[:validation_count]
        class_train_positions = class_positions[validation_count:]

        val_paths.extend(paths_array[class_val_positions].tolist())
        val_labels.extend([class_index] * len(class_val_positions))
        train_paths.extend(paths_array[class_train_positions].tolist())
        train_labels.extend([class_index] * len(class_train_positions))

    return train_paths, train_labels, val_paths, val_labels, class_names

def resolve_dataset(quick: bool) -> tuple[list[str], list[int], list[str], list[int], list[str]]:
    if quick or not SVM_DIR.exists():
        if not SAMPLES_DIR.exists():
            raise FileNotFoundError(
                "No training data found. Expected PlantVillage SVM split or app/samples."
            )
        print("Using app/samples for training (quick pipeline test).")
        return stratified_directory_split(
            SAMPLES_DIR,
            validation_fraction=0.20,
            seed=42,
            minimum_images=3,
        )

    train_map = parse_svm_mapping(SVM_DIR / "train_mapping.txt")
    test_map = parse_svm_mapping(SVM_DIR / "test_mapping.txt")
    id_to_name = {**test_map, **train_map}

    train_paths, train_labels, class_names = collect_svm_split(SVM_DIR / "train", id_to_name)
    val_paths, val_labels, val_class_names = collect_svm_split(SVM_DIR / "test", id_to_name)

    if class_names != val_class_names:
        raise RuntimeError("Train and validation class names do not match.")

    print(f"PlantVillage SVM split: {len(train_paths)} train / {len(val_paths)} validation images")
    return train_paths, train_labels, val_paths, val_labels, class_names


def make_preprocess_fn():
    return keras.applications.efficientnet.preprocess_input


def build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_model(num_classes: int, trainable_backbone: bool = False) -> keras.Model:
    preprocess_input = make_preprocess_fn()
    augmentation = build_augmentation()

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = trainable_backbone

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=trainable_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="plant_disease_efficientnet")
    return model, base_model


def compute_class_weights(labels: list[int], num_classes: int) -> dict[int, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {}
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 1)
        weights[class_idx] = total / (num_classes * count)
    return weights


def make_dataset(
    paths: list[str],
    labels: list[int],
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 10000), seed=42, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def train_phase(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    learning_rate: float,
    class_weight: dict[int, float] | None,
    checkpoint_path: Path,
) -> keras.callbacks.History:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )


def save_metadata(class_names: list[str]) -> None:
    metadata = {
        "backbone": BACKBONE,
        "image_size": list(IMAGE_SIZE),
        "preprocess": "efficientnet",
        "num_classes": len(class_names),
        "class_names": class_names,
    }
    with open(BASE_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    with open(CHECKPOINT_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels, val_paths, val_labels, class_names = resolve_dataset(args.quick)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    print(f"Training images: {len(train_paths)} | Validation images: {len(val_paths)}")

    train_ds = make_dataset(train_paths, train_labels, args.batch_size, training=True)
    val_ds = make_dataset(val_paths, val_labels, args.batch_size, training=False)
    class_weight = compute_class_weights(train_labels, num_classes)

    best_model_path = CHECKPOINT_DIR / "best_model.keras"
    final_model_path = CHECKPOINT_DIR / "final_model.keras"

    print("\nPhase 1: training classification head (frozen backbone)")
    model, base_model = build_model(num_classes, trainable_backbone=False)
    history_head = train_phase(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs_head,
        learning_rate=1e-3,
        class_weight=class_weight,
        checkpoint_path=best_model_path,
    )

    print("\nPhase 2: fine-tuning top backbone layers")
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - 40)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(1 for layer in model.layers if layer.trainable)
    print(f"Trainable layers after unfreezing: {trainable_count}")

    history_finetune = train_phase(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs_finetune,
        learning_rate=1e-5,
        class_weight=class_weight,
        checkpoint_path=best_model_path,
    )

    val_acc = max(
        history_head.history.get("val_accuracy", [0.0]) + history_finetune.history.get("val_accuracy", [0.0])
    )
    print(f"\nBest validation accuracy: {val_acc:.4f}")

    model.save(final_model_path)
    save_metadata(class_names)

    print(f"Saved {best_model_path.name}")
    print(f"Saved {final_model_path.name}")
    print("Updated class_names.json and checkpoints/model_config.json")


if __name__ == "__main__":
    main()
