![Aerial view of crosswalks used for pedestrian path classification](docs/assets/readme-hero.png)

# Pedestrian Path Image Classification

Deep learning project for binary image classification: detecting whether an
image contains a pedestrian path.

The project contains a reproducible TensorFlow/Keras workflow for dataset
splitting, preprocessing, transfer-learning training, fine-tuning, evaluation,
and inference.

Current core settings:

- input size: `250 x 250` RGB images, matching the tile size produced by the
  dataset tooling
- backbone: EfficientNetB0 with ImageNet weights
- default decision threshold: `0.55`
- training augmentation: horizontal flip, light rotation, zoom, translation,
  brightness, contrast, and probabilistic grayscale conversion

The threshold `0.55` is used consistently for evaluation and inference. It is a
slightly stricter positive decision than `0.50`, chosen to reduce false
positives while keeping recall high; the saved threshold analysis should be used
to justify or change this tradeoff for a specific deployment setting.

## Dataset Contribution

The positive class examples were contributed by classmates Sinan, Fabian, Mike,
Joel, Lars, and Neel. The negative class was manually filtered by Frederic
Kurbel from roughly 20,000 candidate images down to the current dataset.

## AI Assistance Disclosure

Codex and ChatGPT were used as assistance tools during project development,
documentation, and review. The project decisions, dataset filtering, and final
responsibility remain with the author.

## Repository Contents

- `00_baseline.py` - compute a majority-class baseline from the training split
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
