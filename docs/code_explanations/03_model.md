# Code Explanation: `03_model.py`

## File Purpose

This file defines the EfficientNetB0 transfer-learning model, compilation logic, custom F1 metric, and model-loading helpers.

## Workflow

1. Create EfficientNetB0 with ImageNet weights and `include_top=False`.
2. Freeze the backbone.
3. Attach a binary classification head.
4. Compile with Adam and Binary Crossentropy.
5. Monitor accuracy, precision, recall, F1-score, and AUC.

## Important Functions and Classes

### `BinaryF1Score`

Purpose: implements F1-score as a Keras metric.

Logic:

```text
F1 = 2 * precision * recall / (precision + recall)
```

Why custom: Keras does not always provide a simple binary F1 metric in all installed versions.

### `build_efficientnetb0_model(...)`

Purpose: builds the complete binary classifier.

Inputs:

- `input_shape`
- `dense_units`
- `dropout_rate`
- `l2_factor`
- `learning_rate`
- `seed`

Output: compiled Keras model.

Architecture:

```text
Input -> EfficientNetB0 -> GlobalAveragePooling2D -> Dense/ReLU -> Dropout -> Dense/Sigmoid
```

### `compile_binary_model(...)`

Purpose: compiles a model with the correct loss, optimizer, and metrics.

Why separated: fine-tuning needs recompilation after changing trainable layers.

### `get_custom_objects()`

Purpose: supplies custom metric classes when loading saved Keras models.

### `predict_label(probability, threshold)`

Purpose: converts sigmoid probability into a human-readable decision.

## Design Reasoning

EfficientNetB0 is used as a pretrained feature extractor. `include_top=False` removes the original ImageNet classifier. Global average pooling reduces parameters and overfitting risk. Sigmoid is appropriate for binary classification.

## Potential Pitfalls

- Forgetting `get_custom_objects()` can break model loading.
- Recompilation is required after changing trainable layers.
- Fine-tuning with too high a learning rate can damage pretrained features.

