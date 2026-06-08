# Pedestrian Path Image Classification

Deep learning project for binary image classification: detecting whether an
image contains a pedestrian path.

The project contains a reproducible TensorFlow/Keras workflow for dataset
splitting, preprocessing, transfer-learning training, fine-tuning, evaluation,
and inference.

## Repository Contents

- `01_split_dataset.py` - create train/validation/test splits
- `02_preprocessing.py` - preprocessing and TensorFlow datasets
- `03_model.py` - model architecture
- `04_train_phase1.py` - initial head training
- `05_fine_tune.py` - fine-tuning
- `06_evaluate_final_model.py` - final evaluation and plots
- `07_inference.py` - inference on new images
- `08_project_metadata.py` - metadata and reproducibility helpers
- `run_workflow.py` - pipeline runner
- `docs/` - full project documentation and code explanations

## Data And Artifacts

Large local files are intentionally excluded from Git:

- `DeepL_Datenset/`
- `dataset/`
- `artifacts/`
- trained model files such as `*.keras` and `*.h5`

The expected local dataset structure is documented in
[`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md).

## Documentation

Start with [`docs/README.md`](docs/README.md), then read the full methodology in
[`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md).
