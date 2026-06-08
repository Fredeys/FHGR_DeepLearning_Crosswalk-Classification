#!/usr/bin/env python3
"""Create a reproducible train/validation/test split for the image dataset.

The test split created by this script is reserved for final model evaluation.
Do not use it for model selection, preprocessing decisions, or hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import config

VALID_IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS
CLASS_FOLDERS = config.ORIGINAL_CLASS_FOLDERS
SPLIT_RATIOS = {
    "train": config.TRAIN_SPLIT,
    "val": config.VAL_SPLIT,
    "test": config.TEST_SPLIT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split DeepL_Datenset into reproducible train/val/test folders."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=config.ORIGINAL_DATASET_DIR,
        help="Path to the original dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATASET_DIR,
        help="Path where the split dataset should be created.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=config.DATASET_SPLIT_MANIFEST,
        help="CSV file documenting source path, class label, split, and target path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Fixed random seed used for deterministic splitting.",
    )
    return parser.parse_args()


def find_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def allocate_split_counts(total: int) -> dict[str, int]:
    """Allocate counts using the largest-remainder method for stable 70/15/15 splits."""
    raw_counts = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: int(count) for split, count in raw_counts.items()}
    remaining = total - sum(counts.values())

    remainders = sorted(
        raw_counts,
        key=lambda split: (raw_counts[split] - counts[split], split),
        reverse=True,
    )
    for split in remainders[:remaining]:
        counts[split] += 1

    return counts


def split_class_images(images: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled_images = images.copy()
    random.Random(seed).shuffle(shuffled_images)

    counts = allocate_split_counts(len(shuffled_images))
    train_end = counts["train"]
    val_end = train_end + counts["val"]

    return {
        "train": shuffled_images[:train_end],
        "val": shuffled_images[train_end:val_end],
        "test": shuffled_images[val_end:],
    }


def warn_if_output_contains_files(output_dir: Path) -> None:
    existing_files = []
    for split in SPLIT_RATIOS:
        for class_label in CLASS_FOLDERS:
            split_class_dir = output_dir / split / class_label
            if split_class_dir.exists():
                existing_files.extend(path for path in split_class_dir.iterdir() if path.is_file())

    if existing_files:
        print(
            f"WARNING: {output_dir} already contains {len(existing_files)} file(s). "
            "This script copies/overwrites matching filenames but does not delete stale files."
        )


def prepare_output_dirs(output_dir: Path) -> None:
    for split in SPLIT_RATIOS:
        for class_label in CLASS_FOLDERS:
            (output_dir / split / class_label).mkdir(parents=True, exist_ok=True)


def validate_sources(source_dir: Path) -> dict[str, list[Path]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source dataset path is not a directory: {source_dir}")

    images_by_class = {}
    for class_label, folder_name in CLASS_FOLDERS.items():
        class_dir = source_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing source class folder: {class_dir}")
        if not class_dir.is_dir():
            raise NotADirectoryError(f"Source class path is not a directory: {class_dir}")

        images = find_images(class_dir)
        if not images:
            raise ValueError(
                f"No valid images found in {class_dir}. "
                f"Allowed extensions: {', '.join(sorted(VALID_IMAGE_EXTENSIONS))}"
            )
        images_by_class[class_label] = images

    return images_by_class


def copy_split_images(
    split_assignments: dict[str, dict[str, list[Path]]],
    output_dir: Path,
) -> list[dict[str, str]]:
    manifest_rows = []
    seen_sources = set()

    for class_label, splits in split_assignments.items():
        for split, image_paths in splits.items():
            target_dir = output_dir / split / class_label
            for source_path in image_paths:
                if source_path in seen_sources:
                    raise RuntimeError(f"Image assigned to multiple splits: {source_path}")
                seen_sources.add(source_path)

                target_path = target_dir / source_path.name
                shutil.copy2(source_path, target_path)
                manifest_rows.append(
                    {
                        "source_path": str(source_path),
                        "class_label": class_label,
                        "split": split,
                        "target_path": str(target_path),
                    }
                )

    return manifest_rows


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["source_path", "class_label", "split", "target_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    counts = defaultdict(int)
    for row in rows:
        counts[(row["split"], row["class_label"])] += 1

    print("\nFinal split counts:")
    for split in SPLIT_RATIOS:
        print(f"{split}:")
        for class_label in CLASS_FOLDERS:
            print(f"  {class_label}: {counts[(split, class_label)]}")


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()

    images_by_class = validate_sources(source_dir)
    warn_if_output_contains_files(output_dir)
    prepare_output_dirs(output_dir)

    split_assignments = {}
    for class_label, images in images_by_class.items():
        class_seed = args.seed + sum(ord(char) for char in class_label)
        split_assignments[class_label] = split_class_images(images, class_seed)

    manifest_rows = copy_split_images(split_assignments, output_dir)
    write_manifest(manifest_path, manifest_rows)
    print_summary(manifest_rows)
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
