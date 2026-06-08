#!/usr/bin/env python3
"""Run inference on a new folder of images.

The script uses the same deterministic image loading convention as training:
RGB images, explicit configured-size bilinear resizing, and pixel values kept in the
0-255 range expected by Keras EfficientNetB0 preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import shutil
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

DEFAULT_THRESHOLD = model_module.DEFAULT_THRESHOLD
get_custom_objects = model_module.get_custom_objects

IMAGE_SIZE = preprocessing.IMAGE_SIZE
VALID_IMAGE_EXTENSIONS = preprocessing.VALID_IMAGE_EXTENSIONS


DEFAULT_MODEL_PATH = config.PHASE2_OUTPUT_DIR / "best_model.keras"
DEFAULT_INFERENCE_DIR = config.INFERENCE_DIR
DEFAULT_LOGS_DIR = config.LOGS_DIR
DEFAULT_ALL_PREDICTIONS = DEFAULT_LOGS_DIR / "inference_predictions.csv"
DEFAULT_POSITIVE_PREDICTIONS = DEFAULT_LOGS_DIR / "positive_predictions.csv"
DEFAULT_POSITIVE_IMAGE_DIR = DEFAULT_INFERENCE_DIR / "positive_images"
DEFAULT_NO_GLOBAL_DIR = config.NO_GLOBAL_INPUT_DIR
DEFAULT_NO_GLOBAL_OUTPUT_DIR = config.NO_GLOBAL_OUTPUT_DIR
DEFAULT_NO_GLOBAL_ALL_PREDICTIONS = DEFAULT_NO_GLOBAL_OUTPUT_DIR / "no_global_predictions.csv"
DEFAULT_NO_GLOBAL_POSITIVE_PREDICTIONS = DEFAULT_NO_GLOBAL_OUTPUT_DIR / "no_global_positive_predictions.csv"
DEFAULT_NO_GLOBAL_POSITIVE_IMAGE_DIR = DEFAULT_NO_GLOBAL_OUTPUT_DIR / "positive_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pedestrian-path presence in a new image folder.")
    parser.add_argument("image_folder", type=Path, help="Folder containing new images for inference.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=config.INFERENCE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_ALL_PREDICTIONS)
    parser.add_argument("--positive-csv", type=Path, default=DEFAULT_POSITIVE_PREDICTIONS)
    parser.add_argument("--positive-image-dir", type=Path, default=DEFAULT_POSITIVE_IMAGE_DIR)
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only read images directly inside image_folder.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_no_global_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external inference on DeepL_Datenset/no_global negative images."
    )
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_NO_GLOBAL_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=config.INFERENCE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_NO_GLOBAL_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_images(image_folder: Path, recursive: bool = True) -> list[Path]:
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")
    if not image_folder.is_dir():
        raise NotADirectoryError(f"Image folder path is not a directory: {image_folder}")

    iterator = image_folder.rglob("*") if recursive else image_folder.iterdir()
    image_paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(
            f"No valid images found in {image_folder}. "
            f"Allowed extensions: {', '.join(sorted(VALID_IMAGE_EXTENSIONS))}"
        )
    return image_paths


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


def predict_probabilities(model: tf.keras.Model, image_paths: list[Path], batch_size: int) -> np.ndarray:
    probabilities = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = load_image_batch(batch_paths)
        batch_probs = model.predict(batch_images, verbose=0).reshape(-1)
        probabilities.extend(float(value) for value in batch_probs)
    return np.asarray(probabilities, dtype=np.float32)


def build_prediction_rows(
    image_paths: list[Path],
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for path, probability in zip(image_paths, probabilities):
        predicted_label = "positive" if probability >= threshold else "negative"
        rows.append(
            {
                "image_path": str(path.resolve()),
                "filename": path.name,
                "prediction_probability": float(probability),
                "predicted_label": predicted_label,
                "threshold": float(threshold),
            }
        )
    return rows


def write_prediction_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "filename",
                "prediction_probability",
                "predicted_label",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def copy_positive_images(rows: list[dict[str, object]], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_count = 0
    for row in rows:
        if row["predicted_label"] != "positive":
            continue

        source_path = Path(str(row["image_path"]))
        target_name = f"{positive_count + 1:04d}_{source_path.name}"
        shutil.copy2(source_path, output_dir / target_name)
        positive_count += 1
    return positive_count


def plot_positive_inference_grid(
    rows: list[dict[str, object]],
    output_path: Path,
    max_cols: int = 4,
) -> None:
    """Save a static grid of all positive predictions."""
    positive_rows = [row for row in rows if row["predicted_label"] == "positive"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not positive_rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.set_title("Positive Inference Predictions")
        ax.text(0.5, 0.5, "No positive predictions", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return

    cols = min(max_cols, len(positive_rows))
    rows_count = int(np.ceil(len(positive_rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.2, rows_count * 3.5))
    axes = np.asarray(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for axis, row in zip(axes, positive_rows):
        image_path = Path(str(row["image_path"]))
        image = tf.keras.utils.load_img(image_path, color_mode=config.COLOR_MODE, target_size=IMAGE_SIZE)
        axis.imshow(image)
        axis.set_title(
            f"{row['filename']}\np={float(row['prediction_probability']):.3f}",
            fontsize=9,
        )

    fig.suptitle("Positive Inference Predictions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_prediction_distribution(rows: list[dict[str, object]], output_path: Path, threshold: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = [float(row["prediction_probability"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(probabilities, bins=30, color="#4477aa", edgecolor="white", alpha=0.9)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.4, label=f"threshold={threshold:.2f}")
    ax.set_title("no_global Prediction Distribution")
    ax.set_xlabel("Predicted probability for pedestrian path")
    ax.set_ylabel("Image count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_inference(
    image_folder: Path,
    model_path: Path,
    threshold: float,
    batch_size: int,
    output_csv: Path,
    positive_csv: Path,
    positive_image_dir: Path,
    grid_path: Path,
    distribution_path: Path | None = None,
    recursive: bool = True,
    dry_run: bool = False,
) -> tuple[list[dict[str, object]], int]:
    image_paths = list_images(image_folder.resolve(), recursive=recursive)

    if dry_run:
        print(f"Dry run complete. Found {len(image_paths)} valid images. No model was loaded.")
        return [], 0

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final model not found: {model_path}. Run 05_fine_tune.py before inference."
        )

    model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())
    probabilities = predict_probabilities(model, image_paths, batch_size)
    rows = build_prediction_rows(image_paths, probabilities, threshold)
    positive_rows = [row for row in rows if row["predicted_label"] == "positive"]

    write_prediction_csv(output_csv, rows)
    write_prediction_csv(positive_csv, positive_rows)
    copied_count = copy_positive_images(positive_rows, positive_image_dir)
    plot_positive_inference_grid(rows, grid_path)
    if distribution_path is not None:
        plot_prediction_distribution(rows, distribution_path, threshold)

    return rows, copied_count


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.image_folder.resolve(), recursive=not args.non_recursive)
    if args.dry_run:
        print(f"Dry run complete. Found {len(image_paths)} valid images. No model was loaded.")
        return

    rows, copied_count = run_inference(
        image_folder=args.image_folder,
        model_path=args.model_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
        positive_csv=args.positive_csv,
        positive_image_dir=args.positive_image_dir,
        grid_path=args.inference_dir / "positive_inference_grid.png",
        recursive=not args.non_recursive,
        dry_run=False,
    )
    positive_rows = [row for row in rows if row["predicted_label"] == "positive"]

    print("Inference complete:")
    print(f"  images processed: {len(rows)}")
    print(f"  threshold: {args.threshold}")
    print(f"  positive predictions: {len(positive_rows)}")
    print(f"  all predictions: {args.output_csv}")
    print(f"  positive predictions: {args.positive_csv}")
    print(f"  copied positive images: {copied_count} -> {args.positive_image_dir}")
    print(f"  positive inference grid: {args.inference_dir / 'positive_inference_grid.png'}")


def no_global_main() -> None:
    args = parse_no_global_args()
    output_dir = args.output_dir.resolve()
    all_csv = output_dir / "no_global_predictions.csv"
    positive_csv = output_dir / "no_global_positive_predictions.csv"
    positive_image_dir = output_dir / "positive_predictions"
    distribution_path = output_dir / "no_global_prediction_distribution.png"
    grid_path = output_dir / "no_global_positive_grid.png"

    # no_global is assumed to contain only negatives, so it stays out of
    # training, validation, test splitting, and first-run hyperparameter
    # decisions. Positive predictions here are useful after training because
    # they reveal likely false positives and candidate hard negatives for a
    # future, separately documented mining iteration.
    rows, copied_count = run_inference(
        image_folder=args.image_folder,
        model_path=args.model_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        output_csv=all_csv,
        positive_csv=positive_csv,
        positive_image_dir=positive_image_dir,
        grid_path=grid_path,
        distribution_path=distribution_path,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    positive_count = sum(row["predicted_label"] == "positive" for row in rows)
    negative_count = len(rows) - positive_count
    positive_percentage = 100.0 * positive_count / max(len(rows), 1)

    print("no_global external inference complete:")
    print(f"  total no_global images processed: {len(rows)}")
    print(f"  predicted negative: {negative_count}")
    print(f"  predicted positive: {positive_count}")
    print(f"  percentage predicted positive: {positive_percentage:.2f}%")
    print(f"  all predictions CSV: {all_csv}")
    print(f"  positive predictions CSV: {positive_csv}")
    print(f"  copied positive predictions: {copied_count} -> {positive_image_dir}")
    print(f"  prediction distribution: {distribution_path}")
    print(f"  positive prediction grid: {grid_path}")


if __name__ == "__main__":
    if Path(__file__).name == "07_inference_no_global.py":
        no_global_main()
    else:
        main()
