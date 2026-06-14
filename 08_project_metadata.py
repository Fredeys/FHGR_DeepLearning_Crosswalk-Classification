"""Reproducibility helpers for the pedestrian-path classification project."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import tensorflow as tf

import config

preprocessing = importlib.import_module("02_preprocessing")
CLASS_NAMES = config.CLASS_NAMES
count_images = preprocessing.count_images
VALID_IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(project_root: Path = config.PROJECT_ROOT) -> dict[str, str | bool | None]:
    """Return the current Git commit and dirty state when the repo is available."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False, "commit": None, "dirty": None}

    return {"available": True, "commit": commit, "dirty": dirty}


def get_split_counts(dataset_dir: Path) -> dict[str, dict[str, int]]:
    split_counts = {}
    for split in ("train", "val", "test"):
        split_counts[split] = {
            class_name: count_images(dataset_dir / split / class_name)
            for class_name in CLASS_NAMES
        }
    return split_counts


def iter_split_images(dataset_dir: Path):
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_dir = dataset_dir / split / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    yield split, class_name, path


def write_image_hash_manifest(output_path: Path, dataset_dir: Path) -> dict[str, Any]:
    """Write per-image SHA-256 hashes for the split dataset and summarize them."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    aggregate = hashlib.sha256()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["split", "class_label", "relative_path", "sha256"],
        )
        writer.writeheader()
        for split, class_name, path in iter_split_images(dataset_dir):
            image_hash = file_sha256(path)
            relative_path = path.relative_to(dataset_dir)
            writer.writerow(
                {
                    "split": split,
                    "class_label": class_name,
                    "relative_path": str(relative_path),
                    "sha256": image_hash,
                }
            )
            aggregate.update(f"{relative_path}:{image_hash}\n".encode("utf-8"))
            row_count += 1

    return {
        "path": str(output_path.resolve()),
        "exists": True,
        "rows": row_count,
        "sha256": file_sha256(output_path),
        "aggregate_dataset_sha256": aggregate.hexdigest(),
    }


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
    image_hash_manifest = write_image_hash_manifest(
        output_path.parent / "image_hashes.csv",
        dataset_dir,
    )
    metadata = {
        "stage": stage,
        "random_seed": seed,
        "git": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "class_names": CLASS_NAMES,
        "dataset_dir": str(dataset_dir.resolve()),
        "split_counts": get_split_counts(dataset_dir),
        "split_manifest": get_manifest_summary(manifest_path),
        "image_hash_manifest": image_hash_manifest,
        "hyperparameters": hyperparameters,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)
