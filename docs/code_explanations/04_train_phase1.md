# Code Explanation: `04_train_phase1.py`

## File Purpose

This script performs phase-1 training: only the classifier head is trained while EfficientNetB0 remains frozen.

## Workflow

1. Load train and validation pipelines.
2. Compute class weights from the training split.
3. Build the frozen-backbone model.
4. Train with callbacks.
5. Save best model, weights, histories, plots, and metadata.

## Important Functions

### `compute_class_weights(train_dir)`

Purpose: handles class imbalance.

Formula:

```text
weight_c = total_samples / (num_classes * samples_in_class_c)
```

Why training only: validation and test data must not influence training.

### `make_callbacks(output_dir)`

Creates:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau
- LearningRateHistory

### `LearningRateHistory`

Purpose: records the learning rate after each epoch so the learning-rate schedule can be plotted.

### `save_history(...)`

Saves training history as JSON and CSV.

### `plot_history(...)`

Saves general training curves.

### `save_training_monitoring_plots(...)`

Saves required PNGs:

- `training_loss.png`
- `training_accuracy.png`
- `learning_rate_schedule.png`
- `precision_recall_training.png`

## Inputs and Outputs

Inputs:

- `dataset/train`
- `dataset/val`

Outputs:

- `artifacts/phase1_head_training/best_model.keras`
- `artifacts/phase1_head_training/phase1_final.weights.h5`
- `artifacts/plots/*.png`
- `artifacts/logs/phase1_training_history.*`

## Design Reasoning

Training only the head is stable and protects pretrained features. Validation loss is used for model selection because it measures generalization better than training loss.

## Potential Pitfalls

- The test set must not be used here.
- If the head does not train, check that the backbone is frozen but the head is trainable.

