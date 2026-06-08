#!/usr/bin/env python3
"""TensorFlow/Keras data pipelines for EfficientNetB0 training.

Images are resized explicitly to the configured image size with bilinear interpolation and are
kept in the 0-255 pixel range. EfficientNetB0 performs its expected
preprocessing internally, so this module does not apply manual 0-1 scaling or
custom mean/std normalization.

The test pipeline is only for final model evaluation.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import tensorflow as tf

import config

IMAGE_SIZE = config.IMAGE_SIZE
BATCH_SIZE = config.BATCH_SIZE
SEED = config.RANDOM_SEED
CLASS_NAMES = config.CLASS_NAMES
VALID_IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS


@tf.keras.utils.register_keras_serializable(package="DeepLProject")
class RandomGrayscale(tf.keras.layers.Layer):
    """Convert a random subset of RGB training images to 3-channel grayscale."""

    def __init__(self, probability: float = 0.15, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.seed = seed
        self.seed_generator = tf.keras.random.SeedGenerator(seed) if seed is not None else None

    def call(self, images, training=None):
        if not training or self.probability <= 0.0:
            return images

        grayscale = tf.image.rgb_to_grayscale(images)
        grayscale = tf.image.grayscale_to_rgb(grayscale)
        batch_size = tf.shape(images)[0]
        random_values = tf.keras.random.uniform(
            shape=(batch_size, 1, 1, 1),
            seed=self.seed_generator,
            dtype=images.dtype,
        )
        use_grayscale = random_values < tf.cast(self.probability, images.dtype)
        return tf.where(use_grayscale, grayscale, images)

    def get_config(self) -> dict:
        layer_config = super().get_config()
        layer_config.update({"probability": self.probability, "seed": self.seed})
        return layer_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation/test data pipelines."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.DATASET_DIR,
        help="Path to the split dataset directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of images per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Fixed random seed for reproducible pipelines.",
    )
    return parser.parse_args()


def set_random_seeds(seed: int = SEED) -> None:
    """Set Python and TensorFlow seeds for reproducible data processing."""
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def validate_dataset_structure(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_dir = dataset_dir / split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing dataset folder: {class_dir}")
            if not class_dir.is_dir():
                raise NotADirectoryError(f"Dataset path is not a folder: {class_dir}")

            image_count = count_images(class_dir)
            if image_count == 0:
                raise ValueError(f"No valid image files found in: {class_dir}")


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def make_augmentation(seed: int = SEED) -> tf.keras.Sequential:
    """Create moderate augmentation used only for the training pipeline.

    The transformations are intentionally mild because overly aggressive crops,
    distortions, erasing, mixup, blur, or artificial artifacts can change the
    semantic structure of pedestrian paths.
    """
    layers = [
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(config.AUGMENT_ROTATION_FACTOR, fill_mode="nearest", seed=seed + 1),
        tf.keras.layers.RandomZoom(config.AUGMENT_ZOOM_FACTOR, fill_mode="nearest", seed=seed + 2),
        tf.keras.layers.RandomTranslation(
            height_factor=config.AUGMENT_TRANSLATION_FACTOR,
            width_factor=config.AUGMENT_TRANSLATION_FACTOR,
            fill_mode="nearest",
            seed=seed + 3,
        ),
        tf.keras.layers.RandomBrightness(
            config.AUGMENT_BRIGHTNESS_FACTOR,
            value_range=(0.0, 255.0),
            seed=seed + 4,
        ),
        tf.keras.layers.RandomContrast(config.AUGMENT_CONTRAST_FACTOR, seed=seed + 5),
    ]
    if config.USE_GRAYSCALE_AUGMENTATION:
        # Grayscale is probabilistic, not global: most batches keep color cues,
        # while some samples force the model to rely on structure, edges, path
        # geometry, and shape instead of memorizing color patterns.
        layers.append(
            RandomGrayscale(
                probability=config.GRAYSCALE_AUGMENTATION_PROBABILITY,
                seed=seed + 6,
                name="random_grayscale",
            )
        )

    return tf.keras.Sequential(layers, name="train_data_augmentation")


def load_split_dataset(
    split_dir: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode=config.COLOR_MODE,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=seed if shuffle else None,
        interpolation=config.INTERPOLATION_METHOD,
    )


def configure_for_performance(dataset: tf.data.Dataset) -> tf.data.Dataset:
    options = tf.data.Options()
    options.deterministic = True
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)


def create_data_pipelines(
    dataset_dir: Path | str = config.DATASET_DIR,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return train, validation, and test datasets for EfficientNetB0.

    Train uses data augmentation. Validation and test are deterministic and do
    not use augmentation. External no_global negatives are deliberately excluded
    from split creation and model selection so they remain an unbiased
    robustness check after training.
    """
    dataset_dir = Path(dataset_dir)
    set_random_seeds(seed)
    validate_dataset_structure(dataset_dir)

    train_ds = load_split_dataset(
        dataset_dir / "train",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = load_split_dataset(
        dataset_dir / "val",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_ds = load_split_dataset(
        dataset_dir / "test",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    augmentation = make_augmentation(seed)
    train_ds = train_ds.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    return (
        configure_for_performance(train_ds),
        configure_for_performance(val_ds),
        configure_for_performance(test_ds),
    )


def create_train_validation_pipelines(
    dataset_dir: Path | str = config.DATASET_DIR,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Return train and validation datasets for model training.

    The test split is intentionally not loaded here because it must remain
    untouched until the final evaluation. no_global is also not loaded here; it
    is reserved for post-training inference and possible later hard-negative
    mining analysis.
    """
    dataset_dir = Path(dataset_dir)
    set_random_seeds(seed)
    validate_dataset_structure(dataset_dir)

    train_ds = load_split_dataset(
        dataset_dir / "train",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = load_split_dataset(
        dataset_dir / "val",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    augmentation = make_augmentation(seed)
    train_ds = train_ds.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    return configure_for_performance(train_ds), configure_for_performance(val_ds)


def print_dataset_summary(dataset_dir: Path) -> None:
    print("Dataset summary:")
    for split in ("train", "val", "test"):
        print(f"{split}:")
        for class_name in CLASS_NAMES:
            print(f"  {class_name}: {count_images(dataset_dir / split / class_name)}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    train_ds, val_ds, test_ds = create_data_pipelines(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print_dataset_summary(dataset_dir)
    print("\nPreprocessing:")
    print(f"  image_size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print("  resizing: bilinear")
    print("  pixel_range: 0-255, no manual normalization")
    print("  train: moderate augmentation enabled")
    print("  augmentation: horizontal flip, light rotation/zoom/shift, light brightness/contrast")
    print(
        "  grayscale_augmentation: "
        f"{config.USE_GRAYSCALE_AUGMENTATION}, probability={config.GRAYSCALE_AUGMENTATION_PROBABILITY}"
    )
    print("  validation/test: deterministic, no augmentation")

    for name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        images, labels = next(iter(dataset))
        print(f"  {name}_batch: images={images.shape}, labels={labels.shape}")


if __name__ == "__main__":
    main()
