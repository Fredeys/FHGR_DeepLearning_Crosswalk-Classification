# Code Explanation: `08_project_metadata.py`

## File Purpose

This helper module saves reproducibility metadata for training, fine-tuning, and evaluation.

## Important Functions

### `file_sha256(path)`

Computes a SHA-256 hash for a file.

Purpose: allows checking whether the split manifest changed.

### `get_git_commit()`

Returns the current Git commit and whether the working tree is dirty when Git
metadata is available.

Purpose: links generated experiment metadata to a concrete repository state.

### `write_image_hash_manifest(...)`

Writes `image_hashes.csv` next to the stage metadata. The file contains the
split, class label, relative path, and SHA-256 hash for every image in the
split dataset.

Purpose: detects data changes that would not be visible from split counts alone.

### `get_split_counts(dataset_dir)`

Counts positive and negative images in train, validation, and test splits.

### `get_manifest_summary(manifest_path)`

Returns manifest path, existence flag, row count, and hash.

### `save_reproducibility_metadata(...)`

Saves:

- stage name
- random seed
- Git commit and dirty state
- Python version
- platform
- TensorFlow version
- Keras version
- class names
- split counts
- split manifest summary
- image hash manifest summary
- hyperparameters

## Design Reasoning

Machine learning experiments are sensitive to environment, data, and random seeds. Saving metadata makes the experiment easier to reproduce, audit, and defend academically.

## Potential Pitfalls

- Metadata does not guarantee bit-exact reproducibility on all hardware.
- If the dataset changes after training, old metadata should not be reused without caution.
- The image hash CSV can become large, but it is more auditable than only
  storing aggregate split counts.
