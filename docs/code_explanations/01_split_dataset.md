# Code Explanation: `01_split_dataset.py`

## File Purpose

This script creates a reproducible train/validation/test split from the original image dataset. It maps `DeepL_Datenset/yes` to `positive` and `DeepL_Datenset/no` to `negative`, while ignoring `DeepL_Datenset/no_global`.

## Workflow

1. Validate source folders.
2. Collect valid images with the configured extension `.png`.
3. Shuffle each class deterministically.
4. Split each class into `70/15/15`.
5. Copy files into `dataset/train`, `dataset/val`, and `dataset/test`.
6. Write `dataset_split_manifest.csv`.
7. Print final class counts.

## Important Functions

### `find_images(class_dir)`

Purpose: returns valid image files in a class directory.

Input: a folder path.

Output: sorted list of image paths.

Pitfall: files with unsupported extensions are ignored.

### `allocate_split_counts(total)`

Purpose: computes split counts using the largest-remainder method.

Logic: multiply total by ratios, take integer floors, then distribute remaining samples according to largest fractional remainder.

Why: avoids losing samples due to rounding.

### `split_class_images(images, seed)`

Purpose: deterministically shuffles images and assigns them to train, validation, and test.

Why per-class: preserves class distribution across splits.

### `copy_split_images(...)`

Purpose: copies images to the target split folders and checks that no source image appears in more than one split.

Why copy instead of move: original data remains unchanged.

### `write_manifest(...)`

Purpose: saves source path, class label, split, and target path.

Why: supports reproducibility and leakage checks.

## Inputs and Outputs

Input:

```text
DeepL_Datenset/yes
DeepL_Datenset/no
```

Output:

```text
dataset/
dataset_split_manifest.csv
```

## Design Reasoning

The split is stratified by class so that all splits contain representative positive and negative examples. A fixed seed makes the split reproducible.

## Potential Pitfalls

- Existing target files are not deleted.
- Exact image overlap is prevented, but grouped geographic leakage is not fully solved without group metadata.
