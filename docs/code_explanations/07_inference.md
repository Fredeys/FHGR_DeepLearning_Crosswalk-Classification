# Code Explanation: `07_inference.py`

## File Purpose

This script applies the final trained model to a new folder of images.

## Workflow

1. Accept an image folder path.
2. List valid images.
3. Load the final best model.
4. Apply the same deterministic preprocessing.
5. Predict probabilities.
6. Save all predictions.
7. Save positive predictions.
8. Copy positive images.
9. Create a positive prediction grid.

## Important Functions

### `list_images(image_folder, recursive)`

Finds valid image files. Can search recursively or only directly inside the folder.

### `load_image_batch(paths)`

Applies the same core inference preprocessing as evaluation:

- RGB
- `224 x 224`
- bilinear interpolation
- `0-255` pixel range

### `predict_probabilities(...)`

Runs model predictions in batches.

### `build_prediction_rows(...)`

Creates structured rows containing path, filename, probability, predicted label, and threshold.

### `write_prediction_csv(...)`

Writes CSV outputs.

### `copy_positive_images(...)`

Copies images predicted as positive into `artifacts/inference/positive_images/`.

### `plot_positive_inference_grid(...)`

Saves a static PNG grid of positive predictions.

## Outputs

- `artifacts/logs/inference_predictions.csv`
- `artifacts/logs/positive_predictions.csv`
- `artifacts/inference/positive_images/`
- `artifacts/inference/positive_inference_grid.png`

## Design Reasoning

Inference is separated from evaluation because new data has no labels and should not be confused with test-set reporting.

## Potential Pitfalls

- The final model must exist before inference.
- A threshold that is too low may copy too many false positives.
- Input images should be representative of the training domain.

