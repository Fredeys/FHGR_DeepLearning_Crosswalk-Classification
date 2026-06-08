# Code Explanation: `05_fine_tune.py`

## File Purpose

This script performs phase-2 fine-tuning. It loads the best phase-1 model and unfreezes only the upper EfficientNetB0 layers.

## Workflow

1. Load the best phase-1 model.
2. Keep lower backbone layers frozen.
3. Keep BatchNorm layers frozen.
4. Unfreeze selected upper layers.
5. Recompile with a smaller learning rate.
6. Continue training with callbacks.
7. Save best fine-tuned model and comparison plots.

## Important Functions

### `load_phase1_model(path)`

Loads the saved phase-1 model. Requires custom objects because of the custom F1 metric.

### `configure_fine_tuning(model, unfreeze_last, learning_rate)`

Purpose: controls which backbone layers are trainable.

Logic:

- freeze all backbone layers
- inspect last `unfreeze_last` layers
- unfreeze non-BatchNorm layers
- keep BatchNorm frozen
- recompile with low learning rate

### `best_val_loss(history)`

Returns the lowest validation loss from a history object.

### `plot_phase_comparison(...)`

Plots phase-1 and fine-tuning curves together.

### `save_fine_tuning_comparison_plots(...)`

Saves:

- `fine_tuning_loss_comparison.png`
- `fine_tuning_accuracy_comparison.png`

## Inputs and Outputs

Input:

```text
artifacts/phase1_head_training/best_model.keras
dataset/train
dataset/val
```

Output:

```text
artifacts/phase2_fine_tuning/best_model.keras
artifacts/plots/fine_tuning_loss_comparison.png
artifacts/plots/fine_tuning_accuracy_comparison.png
artifacts/logs/phase2_fine_tuning_history.*
```

## Design Reasoning

Fine-tuning adapts high-level features to pedestrian paths. A low learning rate reduces catastrophic forgetting.

## Potential Pitfalls

- Fine-tuning before phase 1 is invalid.
- Unfreezing too many layers can overfit.
- Updating BatchNorm can destabilize training.

