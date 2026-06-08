#!/usr/bin/env python3
"""Final one-time test evaluation and error analysis.

Workflow position: preprocessing -> training -> fine-tuning -> final evaluation
-> error analysis.

Run this only after model selection is complete. The isolated test split is used
here exactly once for final reporting.
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
import numpy as np
import tensorflow as tf

import config

model_module = importlib.import_module("03_model")
preprocessing = importlib.import_module("02_preprocessing")
project_metadata = importlib.import_module("08_project_metadata")

DEFAULT_THRESHOLD = model_module.DEFAULT_THRESHOLD
get_custom_objects = model_module.get_custom_objects

CLASS_NAMES = preprocessing.CLASS_NAMES
IMAGE_SIZE = preprocessing.IMAGE_SIZE
SEED = preprocessing.SEED
VALID_IMAGE_EXTENSIONS = preprocessing.VALID_IMAGE_EXTENSIONS
set_random_seeds = preprocessing.set_random_seeds

save_reproducibility_metadata = project_metadata.save_reproducibility_metadata


DEFAULT_MODEL_PATH = config.PHASE2_OUTPUT_DIR / "best_model.keras"
DEFAULT_OUTPUT_DIR = config.FINAL_EVALUATION_DIR
DEFAULT_ERROR_ANALYSIS_DIR = config.ERROR_ANALYSIS_DIR
DEFAULT_PLOTS_DIR = config.PLOTS_DIR
DEFAULT_LOGS_DIR = config.LOGS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the final selected model on the test set.")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--error-analysis-dir", type=Path, default=DEFAULT_ERROR_ANALYSIS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=config.EVALUATION_THRESHOLD)
    parser.add_argument("--max-error-images", type=int, default=config.MAX_ERROR_IMAGES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_test_images(dataset_dir: Path) -> list[tuple[Path, int, str]]:
    """List test files deterministically as (path, label_index, class_name)."""
    test_items = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / "test" / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing test class folder: {class_dir}")
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No valid test images found in: {class_dir}")
        test_items.extend((path, label_index, class_name) for path in image_paths)
    return test_items


def load_image_batch(paths: list[Path]) -> np.ndarray:
    images = []
    for path in paths:
        image = tf.keras.utils.load_img(
            path,
            color_mode=config.COLOR_MODE,
            target_size=IMAGE_SIZE,
            interpolation=config.INTERPOLATION_METHOD,
        )
        images.append(tf.keras.utils.img_to_array(image))
    return np.asarray(images, dtype=np.float32)


def predict_probabilities(
    model: tf.keras.Model,
    items: list[tuple[Path, int, str]],
    batch_size: int,
) -> np.ndarray:
    probabilities = []
    for start in range(0, len(items), batch_size):
        batch_paths = [item[0] for item in items[start : start + batch_size]]
        batch_images = load_image_batch(batch_paths)
        batch_probs = model.predict(batch_images, verbose=0).reshape(-1)
        probabilities.extend(float(value) for value in batch_probs)
    return np.asarray(probabilities, dtype=np.float32)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | dict[str, int]]:
    y_pred = (probabilities >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": counts,
    }


def roc_auc_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities)
    sorted_true = y_true[order]
    positives = np.sum(sorted_true == 1)
    negatives = np.sum(sorted_true == 0)
    if positives == 0 or negatives == 0:
        return float("nan")

    tpr = np.cumsum(sorted_true == 1) / positives
    fpr = np.cumsum(sorted_true == 0) / negatives
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapz(tpr, fpr))


def write_predictions_csv(
    output_path: Path,
    items: list[tuple[Path, int, str]],
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "true_label_index",
                "true_label",
                "probability_positive",
                "predicted_label_index",
                "predicted_label",
                "is_correct",
            ],
        )
        writer.writeheader()
        for (path, true_index, true_label), probability in zip(items, probabilities):
            pred_index = int(probability >= threshold)
            writer.writerow(
                {
                    "image_path": str(path),
                    "true_label_index": true_index,
                    "true_label": true_label,
                    "probability_positive": float(probability),
                    "predicted_label_index": pred_index,
                    "predicted_label": CLASS_NAMES[pred_index],
                    "is_correct": pred_index == true_index,
                }
            )


def threshold_analysis(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    plots_dir: Path,
    logs_dir: Path,
) -> list[dict[str, float]]:
    thresholds = np.round(np.linspace(0.0, 1.0, 101), 2)
    rows = [
        {
            key: value
            for key, value in metrics_at_threshold(y_true, probabilities, threshold).items()
            if key != "confusion_matrix"
        }
        for threshold in thresholds
    ]

    with (logs_dir / "threshold_analysis.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["threshold"] for row in rows], [row["precision"] for row in rows], label="precision")
    ax.plot([row["threshold"] for row in rows], [row["recall"] for row in rows], label="recall")
    ax.plot([row["threshold"] for row in rows], [row["f1_score"] for row in rows], label="f1-score")
    ax.axvline(DEFAULT_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"{DEFAULT_THRESHOLD} default")
    ax.set_xlabel("Sigmoid threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold Analysis")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "threshold_analysis.png", dpi=220)
    plt.close(fig)
    return rows


def plot_confusion_matrix(counts: dict[str, int], plots_dir: Path) -> None:
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred negative", "pred positive"])
    ax.set_yticks([0, 1], labels=["true negative", "true positive"])
    ax.set_title("Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_normalized_confusion_matrix(counts: dict[str, int], plots_dir: Path) -> None:
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], labels=["pred negative", "pred positive"])
    ax.set_yticks([0, 1], labels=["true negative", "true positive"])
    ax.set_title("Normalized Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(plots_dir / "normalized_confusion_matrix.png", dpi=220)
    plt.close(fig)


def roc_curve_points(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[np.inf, np.sort(np.unique(probabilities))[::-1], -np.inf]
    positives = max(np.sum(y_true == 1), 1)
    negatives = max(np.sum(y_true == 0), 1)
    tpr = []
    fpr = []
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        counts = confusion_counts(y_true, y_pred)
        tpr.append(counts["tp"] / positives)
        fpr.append(counts["fp"] / negatives)
    return np.asarray(fpr), np.asarray(tpr)


def precision_recall_curve_points(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[1.01, np.sort(np.unique(probabilities))[::-1], -0.01]
    precision_values = []
    recall_values = []
    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, probabilities, threshold)
        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
    return np.asarray(recall_values), np.asarray(precision_values)


def plot_roc_curve(y_true: np.ndarray, probabilities: np.ndarray, auc_score: float, plots_dir: Path) -> None:
    fpr, tpr = roc_curve_points(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random baseline")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=220)
    plt.close(fig)


def plot_precision_recall_curve(y_true: np.ndarray, probabilities: np.ndarray, plots_dir: Path) -> None:
    recall, precision = precision_recall_curve_points(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "precision_recall_curve.png", dpi=220)
    plt.close(fig)


def plot_prediction_distribution(y_true: np.ndarray, probabilities: np.ndarray, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(probabilities[y_true == 1], bins=30, alpha=0.65, label="true positives", density=False)
    ax.hist(probabilities[y_true == 0], bins=30, alpha=0.65, label="true negatives", density=False)
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("Predicted probability for positive class")
    ax.set_ylabel("Image count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "prediction_distribution.png", dpi=220)
    plt.close(fig)


def plot_class_distribution(dataset_dir: Path, plots_dir: Path) -> None:
    splits = ["train", "val", "test"]
    positive_counts = [preprocessing.count_images(dataset_dir / split / "positive") for split in splits]
    negative_counts = [preprocessing.count_images(dataset_dir / split / "negative") for split in splits]
    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, positive_counts, width, label="positive")
    ax.bar(x + width / 2, negative_counts, width, label="negative")
    ax.set_title("Class Distribution by Split")
    ax.set_xlabel("Dataset split")
    ax.set_ylabel("Image count")
    ax.set_xticks(x, labels=splits)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "class_distribution.png", dpi=220)
    plt.close(fig)


def plot_error_grid(
    examples: list[tuple[Path, int, str, float]],
    output_path: Path,
    title: str,
    max_images: int,
    threshold: float,
) -> None:
    selected = examples[:max_images]
    if not selected:
        return

    cols = min(4, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
    axes = np.asarray(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for axis, (path, true_index, true_label, probability) in zip(axes, selected):
        image = tf.keras.utils.load_img(path, color_mode=config.COLOR_MODE, target_size=IMAGE_SIZE)
        axis.imshow(image)
        predicted_label = CLASS_NAMES[int(probability >= threshold)]
        axis.set_title(
            f"p={probability:.3f}\ntrue={true_label}, pred={predicted_label}",
            fontsize=9,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_error_csv(
    output_path: Path,
    examples: list[tuple[Path, int, str, float]],
    threshold: float,
    split: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "filename",
                "true_label",
                "predicted_label",
                "prediction_probability",
                "split",
            ],
        )
        writer.writeheader()
        for path, _true_index, true_label, probability in examples:
            predicted_label = CLASS_NAMES[int(probability >= threshold)]
            writer.writerow(
                {
                    "image_path": str(path.resolve()),
                    "filename": path.name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "prediction_probability": float(probability),
                    "split": split,
                }
            )


def save_error_thumbnails(
    examples: list[tuple[Path, int, str, float]],
    output_dir: Path,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (path, _true_index, true_label, probability) in enumerate(examples, start=1):
        predicted_label = CLASS_NAMES[int(probability >= threshold)]
        image = tf.keras.utils.load_img(path, color_mode=config.COLOR_MODE, target_size=IMAGE_SIZE)
        safe_stem = path.stem.replace(" ", "_")
        thumbnail_name = (
            f"{index:04d}_{safe_stem}_true-{true_label}_pred-{predicted_label}"
            f"_p-{probability:.3f}.jpg"
        )
        tf.keras.utils.save_img(output_dir / thumbnail_name, tf.keras.utils.img_to_array(image))


def save_error_analysis(
    output_dir: Path,
    error_analysis_dir: Path,
    plots_dir: Path,
    logs_dir: Path,
    items: list[tuple[Path, int, str]],
    probabilities: np.ndarray,
    threshold: float,
    max_images: int,
) -> None:
    false_positives = []
    false_negatives = []
    for (path, true_index, true_label), probability in zip(items, probabilities):
        pred_index = int(probability >= threshold)
        entry = (path, true_index, true_label, float(probability))
        if true_index == 0 and pred_index == 1:
            false_positives.append(entry)
        elif true_index == 1 and pred_index == 0:
            false_negatives.append(entry)

    false_positives.sort(key=lambda item: item[3], reverse=True)
    false_negatives.sort(key=lambda item: item[3])

    false_positive_dir = error_analysis_dir / "false_positives"
    false_negative_dir = error_analysis_dir / "false_negatives"
    write_error_csv(error_analysis_dir / "false_positives.csv", false_positives, threshold, split="test")
    write_error_csv(error_analysis_dir / "false_negatives.csv", false_negatives, threshold, split="test")
    write_error_csv(logs_dir / "false_positives.csv", false_positives, threshold, split="test")
    write_error_csv(logs_dir / "false_negatives.csv", false_negatives, threshold, split="test")
    save_error_thumbnails(false_positives, false_positive_dir, threshold)
    save_error_thumbnails(false_negatives, false_negative_dir, threshold)

    plot_error_grid(
        false_positives,
        plots_dir / "false_positives_grid.png",
        "False Positives: predicted pedestrian path, actually negative",
        max_images,
        threshold,
    )
    plot_error_grid(
        false_negatives,
        plots_dir / "false_negatives_grid.png",
        "False Negatives: missed pedestrian path",
        max_images,
        threshold,
    )

    notes = """# Error Analysis Notes

False positives mean the model predicts `positive` (pedestrian path present), but
the test label is `negative`. In this project, this could overstate pedestrian
path availability.

False negatives mean the model predicts `negative`, but the test label is
`positive`. In this project, this means the model misses an existing pedestrian
path.

When reviewing the saved error grids and `test_predictions.csv`, check for
possible dataset bias:

- Lighting: shadows, overexposure, night/twilight scenes, seasonal brightness.
- Perspective: top-down vs oblique views, unusual camera angles, path scale.
- Urban vs rural environments: sidewalks, roadsides, forest paths, plazas.
- Repetitive scenes: many near-duplicate tiles or repeated geographic patterns.

These notes are a qualitative guide; the script surfaces the errors and
probabilities, while the visual interpretation should be done manually.
"""
    with (output_dir / "error_analysis_notes.md").open("w", encoding="utf-8") as notes_file:
        notes_file.write(notes)
    error_analysis_dir.mkdir(parents=True, exist_ok=True)
    with (error_analysis_dir / "error_analysis_notes.md").open("w", encoding="utf-8") as notes_file:
        notes_file.write(notes)


def print_metric_summary(metrics: dict[str, object], roc_auc: float) -> None:
    counts = metrics["confusion_matrix"]
    print("\nFinal test evaluation:")
    print(f"  threshold: {metrics['threshold']}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  f1_score: {metrics['f1_score']:.4f}")
    print(f"  roc_auc: {roc_auc:.4f}")
    print(f"  confusion_matrix: TN={counts['tn']}, FP={counts['fp']}, FN={counts['fn']}, TP={counts['tp']}")
    print("\nProject meaning:")
    print("  false_positive: model predicts pedestrian path present, but label says none")
    print("  false_negative: model misses a labeled pedestrian path")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    error_analysis_dir = args.error_analysis_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    logs_dir = args.logs_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    error_analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    set_random_seeds(args.seed)
    items = list_test_images(dataset_dir)

    if args.dry_run:
        print(f"Dry run complete. Found {len(items)} test images. No model was loaded.")
        return

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Final model not found: {args.model_path}. Run 05_fine_tune.py before final evaluation."
        )

    model = tf.keras.models.load_model(args.model_path, custom_objects=get_custom_objects())
    probabilities = predict_probabilities(model, items, args.batch_size)
    y_true = np.asarray([item[1] for item in items], dtype=int)

    metrics = metrics_at_threshold(y_true, probabilities, args.threshold)
    roc_auc = roc_auc_score(y_true, probabilities)
    metrics["roc_auc"] = roc_auc

    with (logs_dir / "test_metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=2)

    write_predictions_csv(logs_dir / "test_predictions.csv", items, probabilities, args.threshold)
    threshold_analysis(y_true, probabilities, plots_dir, logs_dir)
    plot_confusion_matrix(metrics["confusion_matrix"], plots_dir)
    plot_normalized_confusion_matrix(metrics["confusion_matrix"], plots_dir)
    plot_roc_curve(y_true, probabilities, roc_auc, plots_dir)
    plot_precision_recall_curve(y_true, probabilities, plots_dir)
    plot_prediction_distribution(y_true, probabilities, plots_dir)
    plot_class_distribution(dataset_dir, plots_dir)
    save_error_analysis(
        output_dir,
        error_analysis_dir,
        plots_dir,
        logs_dir,
        items,
        probabilities,
        args.threshold,
        args.max_error_images,
    )
    save_reproducibility_metadata(
        output_dir / "reproducibility_metadata.json",
        dataset_dir=dataset_dir,
        seed=args.seed,
        stage="final_test_evaluation",
        hyperparameters={
            "model_path": str(args.model_path.resolve()),
            "batch_size": args.batch_size,
            "decision_threshold": args.threshold,
            "test_evaluation_runs": 1,
            "note": "The test set should be evaluated only once after final model selection.",
        },
    )

    print_metric_summary(metrics, roc_auc)
    print("\nSaved artifacts:")
    print(f"  metrics: {logs_dir / 'test_metrics.json'}")
    print(f"  predictions: {logs_dir / 'test_predictions.csv'}")
    print(f"  threshold analysis: {logs_dir / 'threshold_analysis.csv'}")
    print(f"  plots: {plots_dir}")
    print(f"  false positive CSV: {error_analysis_dir / 'false_positives.csv'}")
    print(f"  false negative CSV: {error_analysis_dir / 'false_negatives.csv'}")
    print(f"  false positive thumbnails: {error_analysis_dir / 'false_positives'}")
    print(f"  false negative thumbnails: {error_analysis_dir / 'false_negatives'}")
    print(f"  error notes: {output_dir / 'error_analysis_notes.md'}")
    print(f"  reproducibility metadata: {output_dir / 'reproducibility_metadata.json'}")


if __name__ == "__main__":
    main()
