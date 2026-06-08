# Code Explanation: `02_preprocessing.py`

## File Purpose

This file defines deterministic Keras/TensorFlow data pipelines for training, validation, and testing.

## Workflow

1. Validate dataset folders.
2. Load images from directories.
3. Resize all images to `224 x 224`.
4. Apply augmentation only to training batches.
5. Prefetch datasets for performance.

## Important Constants

- `IMAGE_SIZE = (224, 224)`
- `BATCH_SIZE = 32`
- `SEED = 42`
- `CLASS_NAMES = ["negative", "positive"]`

The class order matters because Keras maps labels according to this list: `negative` becomes `0`, `positive` becomes `1`.

## Important Functions

### `set_random_seeds(seed)`

Sets Python and TensorFlow seeds. This supports reproducibility.

### `validate_dataset_structure(dataset_dir)`

Checks that all expected folders exist and contain valid images.

### `count_images(folder)`

Counts supported image files. Used for checks and class distribution reporting.

### `make_augmentation(seed)`

Builds the training-only augmentation pipeline:

- horizontal flip
- slight rotation
- slight zoom
- slight translation
- slight brightness change
- slight contrast change

These transformations are moderate to avoid destroying pedestrian-path semantics.

### `load_split_dataset(...)`

Uses `tf.keras.utils.image_dataset_from_directory` with:

- RGB color mode
- binary labels
- explicit `224 x 224` image size
- bilinear interpolation
- optional shuffle

### `create_data_pipelines(...)`

Returns train, validation, and test datasets. Used for full pipeline checks.

### `create_train_validation_pipelines(...)`

Returns only train and validation datasets. Used during training and fine-tuning so the test set remains untouched.

## Inputs and Outputs

Input:

```text
dataset/train
dataset/val
dataset/test
```

Output: `tf.data.Dataset` objects.

## Design Reasoning

The test set is isolated. Validation and test data are deterministic. Augmentation is only a regularization tool for training.

## Potential Pitfalls

- Changing `CLASS_NAMES` changes label mapping.
- Adding manual normalization would duplicate EfficientNet preprocessing assumptions.

