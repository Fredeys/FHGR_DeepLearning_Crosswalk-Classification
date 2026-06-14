#!/usr/bin/env python3
"""Simple non-learning baselines for the pedestrian-path classifier.

The baseline predicts the majority class observed in the training split for
every validation and test image. It is intentionally simple: its purpose is to
show what the deep model must beat before transfer learning is justified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import config

CLASS_NAMES = config.CLASS_NAMES
VALID_IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

DEFAULT_OUTPUT_DIR = config.ARTIFACTS_DIR / "baseline"
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute majority-class baselines.")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def split_counts(dataset_dir: Path, split: str) -> dict[str, int]:
    return {
        class_name: count_images(dataset_dir / split / class_name)
        for class_name in CLASS_NAMES
    }


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def metrics_from_counts(
    true_counts: dict[str, int],
    predicted_class: str,
) -> dict[str, float | int | dict[str, int] | str]:
    positive_name = config.POSITIVE_CLASS_NAME
    negative_name = config.NEGATIVE_CLASS_NAME
    positives = true_counts[positive_name]
    negatives = true_counts[negative_name]
    total = positives + negatives

    if predicted_class == positive_name:
        tp = positives
        fp = negatives
        fn = 0
        tn = 0
    else:
        tp = 0
        fp = 0
        fn = positives
        tn = negatives

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_score = 2 * precision * recall / max(precision + recall, EPSILON)

    return {
        "predicted_class": predicted_class,
        "total_images": total,
        "positive_images": positives,
        "negative_images": negatives,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def write_metrics_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "predicted_class",
        "total_images",
        "positive_images",
        "negative_images",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            counts = row["confusion_matrix"]
            writer.writerow(
                {
                    "split": row["split"],
                    "predicted_class": row["predicted_class"],
                    "total_images": row["total_images"],
                    "positive_images": row["positive_images"],
                    "negative_images": row["negative_images"],
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1_score"],
                    "tn": counts["tn"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "tp": counts["tp"],
                }
            )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_counts = split_counts(dataset_dir, "train")
    majority_class = max(train_counts, key=train_counts.get)
    rows = []
    for split in ("val", "test"):
        metrics = metrics_from_counts(split_counts(dataset_dir, split), majority_class)
        metrics["split"] = split
        rows.append(metrics)

    report = {
        "baseline_name": "train_majority_class",
        "purpose": "Non-learning baseline that the trained model should beat.",
        "train_counts": train_counts,
        "majority_class_from_train": majority_class,
        "metrics": rows,
    }
    with (output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
    write_metrics_csv(output_dir / "baseline_metrics.csv", rows)

    print("Majority-class baseline:")
    print(f"  majority class from train: {majority_class}")
    for row in rows:
        print(
            f"  {row['split']}: accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1_score']:.4f}"
        )
    print(f"  saved: {output_dir / 'baseline_metrics.json'}")
    print(f"  saved: {output_dir / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
