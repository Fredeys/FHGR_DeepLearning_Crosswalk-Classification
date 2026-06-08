# Code Explanation: `06_evaluate_final_model.py`

## File Purpose

This script performs final one-time test evaluation after all model selection is complete.

## Workflow

1. Load the final best model.
2. Load test images deterministically.
3. Predict sigmoid probabilities.
4. Compute metrics at the selected threshold.
5. Generate evaluation plots.
6. Export false-positive and false-negative tables and thumbnails.
7. Save reproducibility metadata.

## Important Functions

### `list_test_images(dataset_dir)`

Lists all test images with labels. This is intentionally limited to the test split.

### `load_image_batch(paths)`

Loads RGB images with `224 x 224` bilinear resizing and `0-255` values.

### `predict_probabilities(...)`

Runs model inference in batches and returns positive-class probabilities.

### `metrics_at_threshold(...)`

Computes accuracy, precision, recall, F1-score, and confusion counts at a threshold.

### `roc_auc_score(...)`

Computes ranking-based ROC-AUC.

### `threshold_analysis(...)`

Evaluates thresholds from `0.0` to `1.0` and saves threshold metrics.

### `plot_confusion_matrix(...)`

Saves raw confusion matrix.

### `plot_normalized_confusion_matrix(...)`

Saves row-normalized confusion matrix, useful for class-wise error proportions.

### `plot_roc_curve(...)`

Plots false positive rate vs true positive rate and includes ROC-AUC.

### `plot_precision_recall_curve(...)`

Plots precision vs recall.

### `plot_prediction_distribution(...)`

Shows predicted probability distribution for true positives and true negatives.

### `plot_class_distribution(...)`

Shows positive and negative counts for train, validation, and test.

### `save_error_analysis(...)`

Exports false-positive and false-negative CSVs, thumbnails, and image grids.

## Outputs

- `artifacts/plots/confusion_matrix.png`
- `artifacts/plots/normalized_confusion_matrix.png`
- `artifacts/plots/roc_curve.png`
- `artifacts/plots/precision_recall_curve.png`
- `artifacts/plots/threshold_analysis.png`
- `artifacts/plots/prediction_distribution.png`
- `artifacts/plots/class_distribution.png`
- `artifacts/plots/false_positives_grid.png`
- `artifacts/plots/false_negatives_grid.png`
- `artifacts/logs/test_metrics.json`
- `artifacts/logs/test_predictions.csv`
- `artifacts/logs/threshold_analysis.csv`
- `artifacts/error_analysis/false_positives.csv`
- `artifacts/error_analysis/false_negatives.csv`

## Design Reasoning

Evaluation is isolated from training. The test set is only used here to avoid leakage. Static PNGs are used for reproducible reporting.

## Potential Pitfalls

- Running evaluation repeatedly while changing model decisions can indirectly tune on the test set.
- Threshold choice should be justified using validation data or deployment priorities.

