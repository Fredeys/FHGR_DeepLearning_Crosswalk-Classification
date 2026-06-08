#!/usr/bin/env python3
"""Phase 1 training: train only the classifier head of EfficientNetB0.

This script intentionally uses only the train and validation splits. The test
split is not loaded or evaluated here because it is reserved for final model
evaluation after all model-selection decisions are finished.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config

model_module = importlib.import_module("03_model")
preprocessing = importlib.import_module("02_preprocessing")
project_metadata = importlib.import_module("08_project_metadata")

DEFAULT_DENSE_UNITS = model_module.DEFAULT_DENSE_UNITS
DEFAULT_DROPOUT_RATE = model_module.DEFAULT_DROPOUT_RATE
DEFAULT_L2_FACTOR = model_module.DEFAULT_L2_FACTOR
DEFAULT_LEARNING_RATE = model_module.DEFAULT_LEARNING_RATE
build_efficientnetb0_model = model_module.build_efficientnetb0_model
print_model_strategy = model_module.print_model_strategy

BATCH_SIZE = preprocessing.BATCH_SIZE
CLASS_NAMES = preprocessing.CLASS_NAMES
SEED = preprocessing.SEED
count_images = preprocessing.count_images
create_train_validation_pipelines = preprocessing.create_train_validation_pipelines
set_random_seeds = preprocessing.set_random_seeds

save_reproducibility_metadata = project_metadata.save_reproducibility_metadata


DEFAULT_EPOCHS = config.EPOCHS_PHASE1
DEFAULT_OUTPUT_DIR = config.PHASE1_OUTPUT_DIR
DEFAULT_PLOTS_DIR = config.PLOTS_DIR
DEFAULT_LOGS_DIR = config.LOGS_DIR


class LearningRateHistory(tf.keras.callbacks.Callback):
    """Record the optimizer learning rate at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        learning_rate = self.model.optimizer.learning_rate
        logs["learning_rate"] = float(tf.keras.backend.get_value(learning_rate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EfficientNetB0 classification head with the frozen backbone."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.DATASET_DIR,
        help="Path to the split dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model checkpoints, weights, plots, and history.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        choices=(16, 32),
        help="Batch size for phase 1 training. Use 16 or 32.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum epochs for head training. A value around 10-30 is recommended.",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=DEFAULT_DENSE_UNITS,
        help="Number of ReLU units in the optional dense layer. Use 0 to disable.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Moderate dropout rate in the classifier head. Use 0 to disable.",
    )
    parser.add_argument(
        "--l2-factor",
        type=float,
        default=DEFAULT_L2_FACTOR,
        help="Optional light L2 regularization factor in the classifier head. Use 0 to disable.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Explicit Adam learning rate for phase 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Fixed random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the model and inputs, then stop before fitting.",
    )
    return parser.parse_args()


def compute_class_weights(train_dir: Path) -> dict[int, float]:
    """Compute class weights from the training split only."""
    counts = {
        class_name: count_images(train_dir / class_name)
        for class_name in CLASS_NAMES
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Cannot compute class weights from empty classes: {counts}")

    total = sum(counts.values())
    num_classes = len(CLASS_NAMES)
    class_weights = {
        class_index: total / (num_classes * counts[class_name])
        for class_index, class_name in enumerate(CLASS_NAMES)
    }
    print("Class weights computed from training split only:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_index} ({class_name}): {class_weights[class_index]:.4f}")
    return class_weights


def make_callbacks(output_dir: Path) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks that monitor validation loss for model selection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.keras"

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=config.CHECKPOINT_MONITOR,
            mode="min",
            save_best_only=config.SAVE_BEST_ONLY,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.EARLY_STOPPING_MONITOR,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN,
            mode="min",
            verbose=1,
        ),
        LearningRateHistory(),
    ]


def save_history(
    history: tf.keras.callbacks.History,
    output_dir: Path,
    prefix: str = "training_history",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    json_path = output_dir / f"{prefix}.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(history_dict, json_file, indent=2)

    csv_path = output_dir / f"{prefix}.csv"
    metric_names = list(history_dict)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", *metric_names])
        writer.writeheader()
        for epoch_index in range(len(next(iter(history_dict.values()), []))):
            row = {"epoch": epoch_index + 1}
            row.update({metric: history_dict[metric][epoch_index] for metric in metric_names})
            writer.writerow(row)


def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    """Save loss and accuracy curves for train/validation monitoring."""
    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history_dict.get("loss", []), label="train loss")
    axes[0].plot(epochs, history_dict.get("val_loss", []), label="validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history_dict.get("accuracy", []), label="train accuracy")
    axes[1].plot(epochs, history_dict.get("val_accuracy", []), label="validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=220)
    plt.close(fig)


def save_training_monitoring_plots(history: tf.keras.callbacks.History, plots_dir: Path) -> None:
    """Save required phase-1 monitoring plots as high-resolution PNGs."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    def save_line_plot(filename: str, title: str, ylabel: str, series: list[tuple[str, list[float]]]) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, values in series:
            ax.plot(epochs, values, marker="o", linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=220)
        plt.close(fig)

    save_line_plot(
        "training_loss.png",
        "Training vs Validation Loss",
        "Binary crossentropy",
        [("training loss", history_dict.get("loss", [])), ("validation loss", history_dict.get("val_loss", []))],
    )
    save_line_plot(
        "training_accuracy.png",
        "Training vs Validation Accuracy",
        "Accuracy",
        [("training accuracy", history_dict.get("accuracy", [])), ("validation accuracy", history_dict.get("val_accuracy", []))],
    )
    save_line_plot(
        "learning_rate_schedule.png",
        "Learning Rate Schedule",
        "Learning rate",
        [("learning rate", history_dict.get("learning_rate", []))],
    )
    save_line_plot(
        "precision_recall_training.png",
        "Training Precision and Recall",
        "Metric value",
        [("precision", history_dict.get("precision", [])), ("recall", history_dict.get("recall", []))],
    )


def print_training_notes(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    dropout_rate: float,
    l2_factor: float,
) -> None:
    print("\nTraining phase 1:")
    print("  backbone: frozen")
    print("  trainable part: classification head only")
    print(f"  batch_size: {batch_size}")
    print(f"  max_epochs: {epochs}")
    print(f"  optimizer: Adam, learning_rate={learning_rate}")
    print("  loss: Binary Crossentropy")
    print("  optimization: Adam minimizes the loss function, not the metrics")
    print("  metrics: accuracy, precision, recall, f1_score, auc")
    print("  shuffle: enabled only for training data")
    print("  dropout: classifier head only, no additional dropout in EfficientNetB0")
    print(f"  dropout_rate: {dropout_rate}")
    print(f"  l2_regularization_head: {l2_factor if l2_factor > 0 else 'disabled'}")
    print("  batch_normalization: no additional BatchNorm layers")
    print("  train augmentation: moderate flip, rotation, zoom, shift, brightness, contrast")
    print("  avoided augmentation: strong crops, aggressive distortions, erasing, mixup, blur/artifacts")
    print("  validation augmentation: disabled")
    print("  test split: not loaded, not evaluated")
    print("  callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau")


def main() -> None:
    args = parse_args()
    if not 10 <= args.epochs <= 30:
        print("WARNING: phase 1 is usually trained for about 10-30 epochs.")

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    plots_dir = DEFAULT_PLOTS_DIR.resolve()
    logs_dir = DEFAULT_LOGS_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    set_random_seeds(args.seed)
    train_ds, val_ds = create_train_validation_pipelines(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    class_weights = compute_class_weights(dataset_dir / "train")

    dense_units = args.dense_units if args.dense_units > 0 else None
    model = build_efficientnetb0_model(
        dense_units=dense_units,
        dropout_rate=args.dropout_rate,
        l2_factor=args.l2_factor,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print_model_strategy(model)
    print_training_notes(
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.dropout_rate,
        args.l2_factor,
    )
    model.summary()

    if args.dry_run:
        print("\nDry run complete. No training was performed.")
        return

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(output_dir),
    )

    model.save_weights(output_dir / "phase1_final.weights.h5")
    save_history(history, output_dir)
    save_history(history, logs_dir, prefix="phase1_training_history")
    plot_history(history, output_dir)
    save_training_monitoring_plots(history, plots_dir)
    save_reproducibility_metadata(
        output_dir / "reproducibility_metadata.json",
        dataset_dir=dataset_dir,
        seed=args.seed,
        stage="phase1_head_training",
        hyperparameters={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "dense_units": dense_units,
            "dropout_rate": args.dropout_rate,
            "l2_factor": args.l2_factor,
            "backbone": config.BACKBONE_NAME,
            "backbone_trainable": False,
        },
    )

    print("\nSaved artifacts:")
    print(f"  best model: {output_dir / 'best_model.keras'}")
    print(f"  final weights: {output_dir / 'phase1_final.weights.h5'}")
    print(f"  history JSON: {output_dir / 'training_history.json'}")
    print(f"  history CSV: {output_dir / 'training_history.csv'}")
    print(f"  curves: {output_dir / 'training_curves.png'}")
    print(f"  required plots: {plots_dir}")
    print(f"  history logs: {logs_dir}")
    print(f"  reproducibility metadata: {output_dir / 'reproducibility_metadata.json'}")


if __name__ == "__main__":
    main()
