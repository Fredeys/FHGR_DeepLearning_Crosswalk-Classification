# Pedestrian Path Classification Workflow

This project follows a reproducible transfer-learning workflow:

0. `00_baseline.py`
   - Computes a majority-class baseline from the training split.
   - Reports validation and test metrics without using learned image features.
   - Provides a minimum reference that the deep model should clearly beat.

1. `01_split_dataset.py`
   - Creates deterministic train/validation/test splits.
   - Writes `dataset_split_manifest.csv`.

2. `02_preprocessing.py`
   - Builds Keras data pipelines.
   - Uses explicit `250x250` bilinear resizing.
   - Applies moderate augmentation only to the training set.

3. `03_model.py`
   - Defines the EfficientNetB0 Functional API model, binary metrics, and model-loading helpers.

4. `04_train_phase1.py`
   - Trains the classifier head with EfficientNetB0 frozen.
   - Uses class weights computed only from the training set.
   - Saves the best phase-1 model based on validation loss.

5. `05_fine_tune.py`
   - Loads the best phase-1 model.
   - Fine-tunes only the upper EfficientNetB0 layers and the classifier head.
   - Keeps lower layers and BatchNorm layers frozen.
   - Uses a much smaller learning rate.
   - Selects the best fine-tuned model by lowest validation loss.

6. `06_evaluate_final_model.py`
   - Evaluates the final selected model once on the isolated test set.
   - Saves final metrics, confusion matrix, threshold analysis, predictions, and visual error analysis.
   - Saves false-positive and false-negative CSVs plus thumbnails in `artifacts/error_analysis/`.

7. `07_inference.py`
   - Runs the final selected model on a new image folder.
   - Saves all predictions to `artifacts/inference_predictions.csv`.
   - Saves positive predictions to `artifacts/positive_predictions.csv`.
   - Copies images predicted as positive to `artifacts/inference_positive_images/`.

8. `08_project_metadata.py`
   - Saves reproducibility metadata such as seeds, versions, hyperparameters, and split counts.

Static artifacts are organized as:

```text
artifacts/
  plots/
  error_analysis/
  inference/
  logs/
```

All plots are saved as PNG files; structured outputs are saved as CSV or JSON.

Recommended commands:

```bash
/opt/anaconda3/bin/python 01_split_dataset.py
/opt/anaconda3/bin/python 00_baseline.py
/opt/anaconda3/bin/python 04_train_phase1.py --batch-size 32 --epochs 20
/opt/anaconda3/bin/python 05_fine_tune.py --batch-size 32 --epochs 15 --learning-rate 1e-5 --unfreeze-last 30
/opt/anaconda3/bin/python 06_evaluate_final_model.py --threshold 0.55
/opt/anaconda3/bin/python 07_inference.py /path/to/new/images --threshold 0.55
```

The test set is intentionally used only in `06_evaluate_final_model.py`, after all
training, fine-tuning, and validation-based model selection decisions are done.
