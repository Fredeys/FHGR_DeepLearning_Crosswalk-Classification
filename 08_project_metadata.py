"""Reproducibility helpers for the pedestrian-path classification project."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import platform
from pathlib import Path
from typing import Any

import tensorflow as tf

import config

preprocessing = importlib.import_module("02_preprocessing")
CLASS_NAMES = config.CLASS_NAMES
count_images = preprocessing.count_images


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_split_counts(dataset_dir: Path) -> dict[str, dict[str, int]]:
    split_counts = {}
    for split in ("train", "val", "test"):
        split_counts[split] = {
            class_name: count_images(dataset_dir / split / class_name)
            for class_name in CLASS_NAMES
        }
    return split_counts


def get_manifest_summary(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"path": str(manifest_path), "exists": False}

    with manifest_path.open(newline="", encoding="utf-8") as csv_file:
        row_count = sum(1 for _ in csv.DictReader(csv_file))

    return {
        "path": str(manifest_path.resolve()),
        "exists": True,
        "rows": row_count,
        "sha256": file_sha256(manifest_path),
    }


def save_reproducibility_metadata(
    output_path: Path,
    dataset_dir: Path,
    seed: int,
    hyperparameters: dict[str, Any],
    stage: str,
    manifest_path: Path = config.DATASET_SPLIT_MANIFEST,
) -> None:
    """Save environment, split, and hyperparameter metadata for reproducibility."""
    metadata = {
        "stage": stage,
        "random_seed": seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "class_names": CLASS_NAMES,
        "dataset_dir": str(dataset_dir.resolve()),
        "split_counts": get_split_counts(dataset_dir),
        "split_manifest": get_manifest_summary(manifest_path),
        "hyperparameters": hyperparameters,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)
