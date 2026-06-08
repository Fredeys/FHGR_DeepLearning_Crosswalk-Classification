"""Central configuration for the pedestrian-path classification project.

The project uses Keras with the TensorFlow backend and EfficientNetB0 transfer
learning to classify whether an image contains a pedestrian path. Keep values
that are shared across scripts here so preprocessing, training, fine-tuning,
evaluation, error analysis, and inference use the same assumptions.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------------

PROJECT_NAME = "pedestrian_path_classification"
PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
VERBOSE_MODE = 1


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ORIGINAL_DATASET_DIR = PROJECT_ROOT / "DeepL_Datenset"
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOGS_DIR = ARTIFACTS_DIR / "logs"
ERROR_ANALYSIS_DIR = ARTIFACTS_DIR / "error_analysis"
INFERENCE_DIR = ARTIFACTS_DIR / "inference"
DOCS_DIR = PROJECT_ROOT

PHASE1_OUTPUT_DIR = ARTIFACTS_DIR / "phase1_head_training"
PHASE2_OUTPUT_DIR = ARTIFACTS_DIR / "phase2_fine_tuning"
FINAL_EVALUATION_DIR = ARTIFACTS_DIR / "final_evaluation"
DATASET_SPLIT_MANIFEST = PROJECT_ROOT / "dataset_split_manifest.csv"


# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------

CLASS_NAMES = ["negative", "positive"]
NEGATIVE_CLASS_NAME = "negative"
POSITIVE_CLASS_NAME = "positive"

# Original dataset folder names before the train/validation/test split.
ORIGINAL_CLASS_FOLDERS = {
    POSITIVE_CLASS_NAME: "yes",
    NEGATIVE_CLASS_NAME: "no",
}

# The current dataset consists of PNG tiles. Extend this set if new formats are
# added later.
IMAGE_EXTENSIONS = {".png"}
IMAGE_SIZE = (250, 250)
COLOR_MODE = "rgb"
INTERPOLATION_METHOD = "bilinear"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ---------------------------------------------------------------------------
# Preprocessing settings
# ---------------------------------------------------------------------------

USE_DATA_AUGMENTATION = True
AUGMENT_HORIZONTAL_FLIP = True
AUGMENT_ROTATION_FACTOR = 0.05
AUGMENT_ZOOM_FACTOR = 0.10
AUGMENT_TRANSLATION_FACTOR = 0.05
AUGMENT_BRIGHTNESS_FACTOR = 0.08
AUGMENT_CONTRAST_FACTOR = 0.10
USE_GRAYSCALE_AUGMENTATION = True
GRAYSCALE_AUGMENTATION_PROBABILITY = 0.15
SHUFFLE_TRAIN_DATA = True
DETERMINISTIC_VAL_TEST = True


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

FRAMEWORK = "Keras with TensorFlow backend"
MODEL_TYPE = "Transfer Learning"
BACKBONE_NAME = "EfficientNetB0"
PRETRAINED_WEIGHTS = "imagenet"
INCLUDE_TOP = False
INPUT_SHAPE = (*IMAGE_SIZE, 3)
FREEZE_BACKBONE_INITIAL = True
DENSE_UNITS = 128
DROPOUT_RATE = 0.30
L2_FACTOR = 1e-5
OUTPUT_NEURONS = 1
OUTPUT_ACTIVATION = "sigmoid"
CLASSIFICATION_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Training settings: phase 1
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
INITIAL_LEARNING_RATE = 1e-3
LOSS_FUNCTION = "binary_crossentropy"
OPTIMIZER = "adam"
METRICS = ["accuracy", "precision", "recall", "f1_score", "auc"]
USE_CLASS_WEIGHTS = True


# ---------------------------------------------------------------------------
# Callback settings
# ---------------------------------------------------------------------------

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MONITOR = "val_loss"
REDUCE_LR_ENABLED = True
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-6
CHECKPOINT_ENABLED = True
CHECKPOINT_MONITOR = "val_loss"
SAVE_BEST_ONLY = True


# ---------------------------------------------------------------------------
# Fine-tuning settings
# ---------------------------------------------------------------------------

ENABLE_FINE_TUNING = True
FINE_TUNING_EPOCHS = 15
FINE_TUNING_LEARNING_RATE = INITIAL_LEARNING_RATE / 100.0
UNFREEZE_TOP_LAYERS = 30
FINE_TUNING_BATCH_SIZE = BATCH_SIZE


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

EVALUATION_THRESHOLD = CLASSIFICATION_THRESHOLD
SAVE_CONFUSION_MATRIX = True
SAVE_ROC_CURVE = True
SAVE_PRECISION_RECALL_CURVE = True
SAVE_THRESHOLD_ANALYSIS = True
SAVE_PREDICTION_DISTRIBUTION = True


# ---------------------------------------------------------------------------
# Error analysis settings
# ---------------------------------------------------------------------------

SAVE_FALSE_POSITIVES = True
SAVE_FALSE_NEGATIVES = True
MAX_ERROR_IMAGES = 16
COPY_ERROR_IMAGES = True


# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------

ENABLE_INFERENCE = True
INFERENCE_INPUT_DIR = PROJECT_ROOT / "inference_input"
NO_GLOBAL_INPUT_DIR = ORIGINAL_DATASET_DIR / "no_global"
NO_GLOBAL_OUTPUT_DIR = ARTIFACTS_DIR / "no_global"
SAVE_POSITIVE_PREDICTIONS = True
SAVE_INFERENCE_CSV = True
INFERENCE_THRESHOLD = CLASSIFICATION_THRESHOLD


# ---------------------------------------------------------------------------
# Reproducibility settings
# ---------------------------------------------------------------------------

PYTHON_RANDOM_SEED = RANDOM_SEED
NUMPY_RANDOM_SEED = RANDOM_SEED
TENSORFLOW_RANDOM_SEED = RANDOM_SEED
