#!/usr/bin/env python3
"""Phase 2 fine-tuning for the pedestrian-path classifier.

Workflow position: preprocessing -> phase 1 head training -> fine-tuning ->
final evaluation -> error analysis.

Fine-tuning starts from the best phase-1 model, keeps the lower EfficientNetB0
layers frozen, and updates only the upper backbone layers plus the classifier
head with a much smaller learning rate. This adapts high-level representations
to pedestrian paths without aggressively overwriting pretrained ImageNet
features.
"""

from __future__ import annotations

import argparse
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
phase1 = importlib.import_module("04_train_phase1")

DEFAULT_LEARNING_RATE = model_module.DEFAULT_LEARNING_RATE
compile_binary_model = model_module.compile_binary_model
get_custom_objects = model_module.get_custom_objects

BATCH_SIZE = preprocessing.BATCH_SIZE
SEED = preprocessing.SEED
create_train_validation_pipelines = preprocessing.create_train_validation_pipelines
set_random_seeds = preprocessing.set_random_seeds

save_reproducibility_metadata = project_metadata.save_reproducibility_metadata
compute_class_weights = phase1.compute_class_weights
make_callbacks = phase1.make_callbacks
plot_history = phase1.plot_history
save_history = phase1.save_history


DEFAULT_PHASE1_MODEL = config.PHASE1_OUTPUT_DIR / "best_model.keras"
DEFAULT_OUTPUT_DIR = config.PHASE2_OUTPUT_DIR
DEFAULT_PLOTS_DIR = config.PLOTS_DIR
DEFAULT_LOGS_DIR = config.LOGS_DIR
DEFAULT_FINE_TUNE_LR = config.FINE_TUNING_LEARNING_RATE
DEFAULT_EPOCHS = config.FINE_TUNING_EPOCHS
DEFAULT_UNFREEZE_LAST = config.UNFREEZE_TOP_LAYERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the upper EfficientNetB0 layers.")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--phase1-model", type=Path, default=DEFAULT_PHASE1_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, choices=(16, 32))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_FINE_TUNE_LR)
    parser.add_argument(
        "--unfreeze-last",
        type=int,
        default=DEFAULT_UNFREEZE_LAST,
        help="Number of upper EfficientNetB0 layers considered for fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_phase1_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(
            f"Phase-1 best model not found: {path}. Run 04_train_phase1.py before fine-tuning."
        )
    return tf.keras.models.load_model(path, custom_objects=get_custom_objects())


def configure_fine_tuning(model: tf.keras.Model, unfreeze_last: int, learning_rate: float) -> int:
    """Unfreeze only the upper EfficientNetB0 layers while keeping BatchNorm frozen."""
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    candidate_layers = backbone.layers[-unfreeze_last:] if unfreeze_last > 0 else []
    trainable_backbone_layers = 0
    for layer in candidate_layers:
        # BatchNorm layers keep their pretrained moving statistics. This is a
        # common fine-tuning precaution for small or domain-specific datasets.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_backbone_layers += 1

    compile_binary_model(model, learning_rate=learning_rate)
    return trainable_backbone_layers


def load_history_json(path: Path) -> dict[str, list[float]] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def best_val_loss(history: dict[str, list[float]] | None) -> float | None:
    if not history or not history.get("val_loss"):
        return None
    return min(float(value) for value in history["val_loss"])


def plot_phase_comparison(
    phase1_history: dict[str, list[float]] | None,
    phase2_history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Plot phase-1 and fine-tuning curves together for validation comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for metric, axis, title in (
        ("loss", axes[0], "Loss"),
        ("accuracy", axes[1], "Accuracy"),
    ):
        offset = 0
        if phase1_history and phase1_history.get(metric):
            epochs = range(1, len(phase1_history[metric]) + 1)
            axis.plot(epochs, phase1_history[metric], label=f"phase 1 train {metric}")
            axis.plot(epochs, phase1_history.get(f"val_{metric}", []), label=f"phase 1 val {metric}")
            offset = len(phase1_history[metric])

        epochs = range(offset + 1, offset + len(phase2_history.get(metric, [])) + 1)
        axis.plot(epochs, phase2_history.get(metric, []), label=f"fine-tune train {metric}")
        axis.plot(epochs, phase2_history.get(f"val_{metric}", []), label=f"fine-tune val {metric}")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "phase1_vs_finetuning_curves.png", dpi=160)
    plt.close(fig)


def save_fine_tuning_comparison_plots(
    phase1_history: dict[str, list[float]] | None,
    phase2_history: dict[str, list[float]],
    plots_dir: Path,
) -> None:
    """Save separate loss and accuracy comparison PNGs for phase 1 vs fine-tuning."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    def save_metric(metric: str, filename: str, title: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        offset = 0
        if phase1_history and phase1_history.get(metric):
            phase1_epochs = range(1, len(phase1_history[metric]) + 1)
            ax.plot(phase1_epochs, phase1_history[metric], label=f"phase 1 train {metric}")
            ax.plot(phase1_epochs, phase1_history.get(f"val_{metric}", []), label=f"phase 1 val {metric}")
            offset = len(phase1_history[metric])

        phase2_epochs = range(offset + 1, offset + len(phase2_history.get(metric, [])) + 1)
        ax.plot(phase2_epochs, phase2_history.get(metric, []), label=f"fine-tuning train {metric}")
        ax.plot(phase2_epochs, phase2_history.get(f"val_{metric}", []), label=f"fine-tuning val {metric}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=220)
        plt.close(fig)

    save_metric("loss", "fine_tuning_loss_comparison.png", "Phase 1 vs Fine-Tuning Loss", "Binary crossentropy")
    save_metric("accuracy", "fine_tuning_accuracy_comparison.png", "Phase 1 vs Fine-Tuning Accuracy", "Accuracy")


def print_fine_tuning_notes(
    learning_rate: float,
    unfreeze_last: int,
    trainable_backbone_layers: int,
    epochs: int,
    batch_size: int,
) -> None:
    print("\nFine-tuning phase:")
    print("  starting_point: best phase-1 model")
    print("  goal: adapt upper high-level EfficientNetB0 features to pedestrian paths")
    print("  lower_backbone_layers: frozen")
    print("  batch_norm_layers: kept frozen")
    print(f"  upper_layers_considered: {unfreeze_last}")
    print(f"  trainable_backbone_layers: {trainable_backbone_layers}")
    print("  classifier_head: trainable")
    print(f"  learning_rate: {learning_rate} (significantly smaller than phase 1)")
    print(f"  batch_size: {batch_size}")
    print(f"  max_epochs: {epochs}")
    print("  callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau")
    print("  model_selection: lowest validation loss")
    print("  test split: not loaded, not evaluated")


def main() -> None:
    args = parse_args()
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

    model = load_phase1_model(args.phase1_model.resolve())
    trainable_backbone_layers = configure_fine_tuning(
        model=model,
        unfreeze_last=args.unfreeze_last,
        learning_rate=args.learning_rate,
    )
    print_fine_tuning_notes(
        learning_rate=args.learning_rate,
        unfreeze_last=args.unfreeze_last,
        trainable_backbone_layers=trainable_backbone_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model.summary()

    if args.dry_run:
        print("\nDry run complete. No fine-tuning was performed.")
        return

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(output_dir),
    )

    model.save_weights(output_dir / "fine_tuned_final.weights.h5")
    save_history(history, output_dir)
    save_history(history, logs_dir, prefix="phase2_fine_tuning_history")
    plot_history(history, output_dir)

    phase1_history = load_history_json(args.phase1_model.parent / "training_history.json")
    phase2_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    plot_phase_comparison(phase1_history, phase2_history, output_dir)
    save_fine_tuning_comparison_plots(phase1_history, phase2_history, plots_dir)

    phase1_best = best_val_loss(phase1_history)
    phase2_best = best_val_loss(phase2_history)
    summary = {
        "phase1_best_val_loss": phase1_best,
        "fine_tuning_best_val_loss": phase2_best,
        "fine_tuning_improved_validation_loss": (
            phase1_best is not None and phase2_best is not None and phase2_best < phase1_best
        ),
        "selection_rule": "lowest validation loss",
        "best_model_path": str((output_dir / "best_model.keras").resolve()),
    }
    with (output_dir / "fine_tuning_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
    with (logs_dir / "fine_tuning_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    save_reproducibility_metadata(
        output_dir / "reproducibility_metadata.json",
        dataset_dir=dataset_dir,
        seed=args.seed,
        stage="phase2_fine_tuning",
        hyperparameters={
            "phase1_model": str(args.phase1_model.resolve()),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "unfreeze_last": args.unfreeze_last,
            "trainable_backbone_layers": trainable_backbone_layers,
            "batch_norm_trainable": False,
            "model_selection": "lowest validation loss",
        },
    )

    print("\nValidation comparison:")
    print(f"  phase 1 best val_loss: {phase1_best}")
    print(f"  fine-tuning best val_loss: {phase2_best}")
    print(f"  improved: {summary['fine_tuning_improved_validation_loss']}")
    print("\nSaved artifacts:")
    print(f"  best fine-tuned model: {output_dir / 'best_model.keras'}")
    print(f"  final weights: {output_dir / 'fine_tuned_final.weights.h5'}")
    print(f"  history: {output_dir / 'training_history.json'}")
    print(f"  curves: {output_dir / 'training_curves.png'}")
    print(f"  comparison curves: {output_dir / 'phase1_vs_finetuning_curves.png'}")
    print(f"  required comparison plots: {plots_dir}")
    print(f"  logs: {logs_dir}")
    print(f"  reproducibility metadata: {output_dir / 'reproducibility_metadata.json'}")


if __name__ == "__main__":
    main()
