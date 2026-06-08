# Code Explanation: `08_project_metadata.py`

## File Purpose

This helper module saves reproducibility metadata for training, fine-tuning, and evaluation.

## Important Functions

### `file_sha256(path)`

Computes a SHA-256 hash for a file.

Purpose: allows checking whether the split manifest changed.

### `get_split_counts(dataset_dir)`

Counts positive and negative images in train, validation, and test splits.

### `get_manifest_summary(manifest_path)`

Returns manifest path, existence flag, row count, and hash.

### `save_reproducibility_metadata(...)`

Saves:

- stage name
- random seed
- Python version
- platform
- TensorFlow version
- Keras version
- class names
- split counts
- split manifest summary
- hyperparameters

## Design Reasoning

Machine learning experiments are sensitive to environment, data, and random seeds. Saving metadata makes the experiment easier to reproduce, audit, and defend academically.

## Potential Pitfalls

- Metadata does not guarantee bit-exact reproducibility on all hardware.
- If the dataset changes after training, old metadata should not be reused without caution.

