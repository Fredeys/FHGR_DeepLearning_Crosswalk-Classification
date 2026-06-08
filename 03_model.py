#!/usr/bin/env python3
"""EfficientNetB0 transfer-learning model for binary image classification.

The model uses the Keras Functional API so the data flow from input tensor,
through the frozen ImageNet backbone, to the binary classifier head remains
explicit and easy to adapt for later fine-tuning.
"""

from __future__ import annotations

import argparse
import importlib

import tensorflow as tf

import config

preprocessing = importlib.import_module("02_preprocessing")
IMAGE_SIZE = config.IMAGE_SIZE
SEED = config.RANDOM_SEED
set_random_seeds = preprocessing.set_random_seeds


INPUT_SHAPE = config.INPUT_SHAPE
DEFAULT_DENSE_UNITS = config.DENSE_UNITS
DEFAULT_DROPOUT_RATE = config.DROPOUT_RATE
DEFAULT_L2_FACTOR = config.L2_FACTOR
DEFAULT_LEARNING_RATE = config.INITIAL_LEARNING_RATE
DEFAULT_THRESHOLD = config.CLASSIFICATION_THRESHOLD


@tf.keras.utils.register_keras_serializable(package="DeepLProject")
class BinaryF1Score(tf.keras.metrics.Metric):
    """F1-score for binary classification at a fixed sigmoid threshold."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, name: str = "f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self) -> None:
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and summarize the EfficientNetB0 transfer-learning model."
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=DEFAULT_DENSE_UNITS,
        help="Number of ReLU units in the optional dense classifier layer.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Dropout rate used before the sigmoid output layer. Use 0 to disable.",
    )
    parser.add_argument(
        "--l2-factor",
        type=float,
        default=DEFAULT_L2_FACTOR,
        help="Optional light L2 regularization factor for the classifier head. Use 0 to disable.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for the initial frozen-backbone training phase.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Fixed random seed for reproducible model initialization.",
    )
    return parser.parse_args()


def build_efficientnetb0_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    dense_units: int | None = DEFAULT_DENSE_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    l2_factor: float = DEFAULT_L2_FACTOR,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = SEED,
) -> tf.keras.Model:
    """Build a compiled EfficientNetB0 transfer-learning model.

    EfficientNetB0 includes its own preprocessing behavior in Keras, so the
    input tensor expects RGB images in the 0-255 pixel range.
    """
    set_random_seeds(seed)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=config.INCLUDE_TOP,
        weights=config.PRETRAINED_WEIGHTS,
        input_shape=input_shape,
    )
    backbone.trainable = not config.FREEZE_BACKBONE_INITIAL

    head_regularizer = (
        tf.keras.regularizers.l2(l2_factor)
        if l2_factor is not None and l2_factor > 0
        else None
    )

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    if dense_units is not None and dense_units > 0:
        x = tf.keras.layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=head_regularizer,
            name="dense_relu",
        )(x)

    if dropout_rate > 0:
        # Dropout is applied only in the newly trained classifier head. The
        # pretrained EfficientNetB0 backbone is left unchanged.
        x = tf.keras.layers.Dropout(dropout_rate, seed=seed, name="dropout")(x)

    outputs = tf.keras.layers.Dense(
        config.OUTPUT_NEURONS,
        activation=config.OUTPUT_ACTIVATION,
        kernel_regularizer=head_regularizer,
        name="pedestrian_path_probability",
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_transfer")
    compile_binary_model(model, learning_rate=learning_rate)
    return model


def compile_binary_model(
    model: tf.keras.Model,
    learning_rate: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> tf.keras.Model:
    """Compile the binary classifier.

    Adam optimizes Binary Crossentropy. The metrics are monitored for model
    selection and interpretation, but they are not the optimization target.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
            tf.keras.metrics.Precision(name="precision", thresholds=threshold),
            tf.keras.metrics.Recall(name="recall", thresholds=threshold),
            BinaryF1Score(threshold=threshold, name="f1_score"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_custom_objects() -> dict[str, type[BinaryF1Score]]:
    """Return custom objects needed when loading saved Keras models."""
    return {"BinaryF1Score": BinaryF1Score}


def predict_label(probability: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Convert a sigmoid probability into the project-specific class decision."""
    if probability >= threshold:
        return "Fussgaengerweg vorhanden"
    return "kein Fussgaengerweg"


def print_model_strategy(model: tf.keras.Model) -> None:
    backbone = model.get_layer("efficientnetb0")
    print("Model strategy:")
    print(f"  framework: {config.FRAMEWORK}")
    print(f"  method: {config.MODEL_TYPE}")
    print("  api: Keras Functional API")
    print(f"  backbone: {config.BACKBONE_NAME}, {config.PRETRAINED_WEIGHTS} weights, include_top={config.INCLUDE_TOP}")
    print(f"  backbone_trainable_initially: {backbone.trainable}")
    print("  classifier: GlobalAveragePooling2D -> optional Dense/ReLU -> optional Dropout -> sigmoid")
    print("  regularization: moderate head dropout, optional light head L2, no extra backbone dropout")
    print("  batch_normalization: no additional BatchNorm; EfficientNetB0 already contains BatchNorm")
    print("  loss: Binary Crossentropy")
    print("  optimizer: Adam")
    print("  optimization_target: Adam minimizes the loss, not the metrics")
    print("  metrics: accuracy, precision, recall, f1_score, auc")
    print(f"  decision_threshold: {DEFAULT_THRESHOLD}")
    print(f"  p >= {DEFAULT_THRESHOLD}: Fussgaengerweg vorhanden")
    print(f"  p < {DEFAULT_THRESHOLD}: kein Fussgaengerweg")


def main() -> None:
    args = parse_args()
    model = build_efficientnetb0_model(
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        l2_factor=args.l2_factor,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print_model_strategy(model)
    print()
    model.summary()


if __name__ == "__main__":
    main()
