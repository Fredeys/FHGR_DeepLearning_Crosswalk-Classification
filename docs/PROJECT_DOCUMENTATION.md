# Project Documentation: Pedestrian Path Image Classification

## A. Project Overview

### Project Goal

The goal of this project is to build a reproducible deep learning pipeline that classifies whether an image contains a pedestrian path. This is a binary image classification task:

- `positive`: pedestrian path present
- `negative`: no pedestrian path present

The model outputs a sigmoid probability `p` for the positive class. By default, the decision threshold is `0.5`:

- `p >= 0.5`: pedestrian path present
- `p < 0.5`: no pedestrian path present

### Why This Is a Binary Image Classification Problem

Each input image receives exactly one target label from two mutually exclusive classes. The task is therefore not object detection, because the model does not predict bounding boxes. It is also not semantic segmentation, because the model does not produce a pixel-level mask of the path. It answers a global image-level question: does this image contain a pedestrian path?

### Why Deep Learning Was Chosen

Pedestrian paths can vary strongly in color, texture, lighting, perspective, surrounding environment, and scale. Hand-designed image features such as edges, color thresholds, or texture descriptors would be difficult to generalize across urban sidewalks, rural paths, roadsides, shadows, vegetation, and different image qualities. Convolutional neural networks learn hierarchical visual features directly from data, from low-level edges to higher-level object and scene patterns.

### Why Transfer Learning Was Chosen

Training a CNN from scratch usually requires a very large dataset and substantial compute resources. Transfer learning reuses a backbone pretrained on ImageNet. The pretrained network already contains useful low-level and mid-level visual features such as edges, corners, textures, shapes, and object parts. The project only needs to learn how these features relate to the pedestrian-path classification task. This improves data efficiency, reduces training time, and usually gives better generalization for university-scale datasets.

## B. Dataset Documentation

### Dataset Structure

The original dataset is stored as:

```text
DeepL_Datenset/
  yes/
  no/
  no_global/   # ignored
```

The split dataset is stored as:

```text
dataset/
  train/
    positive/
    negative/
  val/
    positive/
    negative/
  test/
    positive/
    negative/
```

The `yes` folder is mapped to `positive`, and the `no` folder is mapped to `negative`.

### Class Distribution

The current split produced:

```text
train: positive 2729, negative 4513
val:   positive 585,  negative 967
test:  positive 585,  negative 967
```

The negative class is larger than the positive class. This imbalance matters because an unweighted model could improve accuracy by favoring the majority class.

### Imbalance Handling and Class Weights

Class weights are computed only from the training split. For class `c`, the script uses:

```text
weight_c = total_training_samples / (number_of_classes * samples_in_class_c)
```

This gives larger weight to the minority class. The loss contribution of positive samples is increased relative to negative samples, discouraging the model from ignoring the less frequent positive class.

Class weights are not computed from validation or test data because those splits must simulate unseen data.

### Train / Validation / Test Strategy

The dataset is split into:

- `70%` training
- `15%` validation
- `15%` test

The training set is used for fitting model parameters. The validation set is used for model selection, early stopping, learning-rate adaptation, and fine-tuning decisions. The test set is used only once after all model choices are fixed.

### Leakage Prevention

Data leakage occurs when information from validation or test data influences training or model selection. This project prevents leakage by:

- copying instead of moving original files
- assigning each image to only one split
- saving a manifest file with every image assignment
- applying augmentation only to the training split
- computing class weights only from the training split
- using the test set only in `06_evaluate_final_model.py`

### Why Stratified Split Was Used

A stratified split preserves the class ratio in train, validation, and test sets. Without stratification, one split might contain too few positive examples, producing unstable validation metrics or misleading test results. Stratification is especially important when classes are imbalanced.

### Why Grouped Split Matters

If several images are near-duplicates or come from the same geographic scene, a random image-level split can put very similar examples into both train and test sets. This can make the model look better than it truly is because the test set is no longer independent. A grouped split would ensure all images from the same location, tile group, or acquisition source remain in the same split. The current implementation prevents exact image overlap, but grouped splitting would be a valuable future improvement if geographic or scene group identifiers are available.

## C. Preprocessing Documentation

### Image Size Choice

All images are explicitly resized to `224 x 224` pixels. EfficientNetB0 is commonly used with this input size, and the pretrained ImageNet weights are compatible with it. Using a fixed size is necessary because neural networks process tensors with consistent dimensions.

### Why 224 x 224 Was Selected

`224 x 224` is a practical compromise:

- large enough to preserve meaningful scene information
- small enough for efficient training
- compatible with EfficientNetB0 defaults
- widely used in transfer-learning workflows

Larger images might preserve more detail but increase memory usage and training time. Smaller images might lose path structure and scene context.

### Resizing Method

Bilinear interpolation is used. It is deterministic, efficient, and produces smooth resized images. Nearest-neighbor interpolation could introduce blocky artifacts, while more expensive interpolation methods are usually unnecessary for this classification task.

### Preprocessing Pipeline

The Keras image loader returns RGB tensors with shape:

```text
(batch_size, 224, 224, 3)
```

Images remain in the `0-255` pixel range. No manual scaling to `0-1` and no custom mean/std normalization are applied because Keras EfficientNetB0 includes its expected preprocessing behavior internally.

### Deterministic Validation and Test Pipelines

Validation and test pipelines are deterministic:

- no shuffling
- no augmentation
- fixed image size
- fixed interpolation

This makes evaluation stable and reproducible.

### Why Augmentation Is Only Used on Training Data

Augmentation artificially increases variation during training and improves robustness. It must not be applied to validation or test data because those splits should measure model performance on real, unmodified data. Applying augmentation to validation or test images would make evaluation stochastic and harder to interpret.

### Augmentation Methods

The training pipeline uses moderate augmentation:

- horizontal flip: helps if left-right orientation is not semantically important
- slight rotation: improves robustness to small orientation changes
- slight zoom: handles scale variation
- slight translation: handles small spatial shifts
- slight brightness change: handles lighting differences
- slight contrast change: handles contrast variation

Aggressive distortions, strong crops, random erasing, mixup, strong blur, and artificial artifacts are avoided because they can destroy or alter the semantic structure of pedestrian paths.

## D. Model Architecture Documentation

### EfficientNetB0

EfficientNet is a family of convolutional neural networks designed to balance depth, width, and input resolution through compound scaling. EfficientNetB0 is the smallest baseline model in the family. It is efficient enough for university hardware while still providing strong feature extraction performance.

EfficientNetB0 is suitable here because:

- it is pretrained and widely supported in Keras
- it has good accuracy-to-parameter efficiency
- it is lighter than many larger CNNs
- it works well for transfer learning

### Transfer Learning

Transfer learning uses knowledge learned from a large source dataset and adapts it to a smaller target task. In this project, ImageNet pretraining provides general visual features, and the pedestrian-path dataset teaches the final classifier how to use those features for the specific binary decision.

### ImageNet Pretraining

ImageNet contains millions of labeled natural images. A model pretrained on ImageNet learns general image representations such as edges, textures, shapes, and object parts. These representations are often useful even when the target task differs from ImageNet classes.

### `include_top=False`

`include_top=False` removes the original ImageNet classification head. The original head predicts 1000 ImageNet classes, which is irrelevant for pedestrian-path classification. Removing it allows the project to attach a custom binary classifier head.

### Frozen Backbone

In phase 1, the EfficientNetB0 backbone is frozen. This means its pretrained weights are not updated. Only the new classification head is trained. This prevents unstable early training and protects useful pretrained features while the new head learns the task.

### Classification Head

The head contains:

```text
GlobalAveragePooling2D
optional Dense(ReLU)
optional Dropout
Dense(1, sigmoid)
```

### GlobalAveragePooling2D

Global average pooling converts the spatial feature map from the backbone into a compact feature vector by averaging each channel. Compared with flattening, it has fewer parameters and reduces overfitting risk.

### Dense Layer

The optional Dense layer with ReLU learns task-specific combinations of backbone features. ReLU introduces non-linearity:

```text
ReLU(x) = max(0, x)
```

### Sigmoid Output

The final layer has one neuron with sigmoid activation:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

It maps the raw logit to a probability-like value between 0 and 1.

### Thresholding

The default threshold is `0.5`. Threshold analysis is implemented because the best threshold depends on the desired tradeoff between precision and recall.

## E. Keras Functional API

The Functional API was selected instead of the Sequential API because transfer learning benefits from explicit control over inputs, backbone, and head. It makes the data flow clearer:

```text
Input -> EfficientNetB0 backbone -> pooling -> dense/dropout -> sigmoid output
```

The Functional API is more flexible for multi-input models, feature extraction, fine-tuning, and model inspection. The Sequential API is simpler but less expressive for architectures involving pretrained submodels.

## F. Training Documentation

### Epochs

An epoch is one complete pass over the training dataset. Phase 1 uses a default range around 10-30 epochs, with `20` as a practical default. EarlyStopping may stop training earlier if validation loss stops improving.

### Batch Size

Batch size controls how many images are processed before one gradient update. The project supports `16` or `32`. Smaller batches use less memory and can add gradient noise. Larger batches are faster but require more memory.

### Learning Rate

The learning rate controls step size during optimization. Phase 1 uses `1e-3` by default. Fine-tuning uses a much smaller learning rate, typically `1e-5`, because pretrained backbone weights should be adjusted carefully.

### Adam Optimizer

Adam combines adaptive learning rates with momentum-like behavior. It is robust, widely used, and works well for transfer learning. Alternatives include SGD with momentum, RMSProp, or AdamW. SGD can generalize well but often requires more careful tuning.

### Binary Crossentropy

Binary crossentropy is the loss function for binary classification:

```text
L = -[y log(p) + (1 - y) log(1 - p)]
```

where `y` is the true label and `p` is the predicted probability for the positive class.

### Loss vs Metrics

The optimizer minimizes the loss function, not the metrics. Metrics such as accuracy, precision, recall, and F1 are monitored for interpretation and model selection, but gradient updates are computed from binary crossentropy.

### Class Weights

Class weights modify the loss so minority-class mistakes matter more. This is important because the dataset has more negative than positive examples.

### Callbacks

#### EarlyStopping

EarlyStopping monitors validation loss and stops training when it stagnates. This saves time and reduces overfitting.

#### ModelCheckpoint

ModelCheckpoint saves the best model according to validation loss. This matters because the final epoch is not always the best epoch.

#### ReduceLROnPlateau

ReduceLROnPlateau lowers the learning rate when validation loss stops improving. This can help the optimizer make smaller, more stable updates later in training.

## G. Overfitting and Regularization

### What Is Overfitting?

Overfitting occurs when a model learns training-specific patterns instead of general patterns. It performs well on training data but poorly on validation or test data.

### Why Overfitting Is Dangerous

The goal is not memorization but generalization. A model that overfits may fail on new geographic regions, lighting conditions, or scene types.

### Dropout

Dropout randomly deactivates neurons during training. This discourages co-adaptation and makes the head more robust. The project uses moderate dropout only in the classifier head, not in the EfficientNetB0 backbone.

### Why Moderate Dropout Was Used

Too little dropout may not regularize enough. Too much dropout can underfit, especially when the head is small. A moderate rate such as `0.3` is a common compromise.

### Why Additional BatchNorm Was Not Added

EfficientNetB0 already contains BatchNormalization layers. Adding extra BatchNorm layers in the small classifier head would increase complexity and is unlikely to provide a clear benefit. During fine-tuning, BatchNorm layers are kept frozen to preserve pretrained statistics.

### Augmentation Strategy

Moderate augmentation improves generalization while preserving semantics. Aggressive augmentation is avoided because it can make pedestrian paths unrealistic or remove important visual cues.

## H. Fine-Tuning

### Why the Backbone Is Initially Frozen

At the beginning, the new classifier head has random weights. If the entire network were trainable immediately, large gradient updates could damage useful pretrained features. Freezing the backbone stabilizes learning.

### Why Upper Layers Are Later Unfrozen

Lower CNN layers learn generic features such as edges and textures. Upper layers learn more task-specific abstract features. Fine-tuning the upper layers adapts the model to pedestrian-path patterns while preserving general low-level visual knowledge.

### Why a Lower Learning Rate Is Required

Fine-tuning updates pretrained weights. A large learning rate could overwrite useful ImageNet features. A small learning rate allows careful adaptation.

### Catastrophic Forgetting

Catastrophic forgetting occurs when a model loses previously learned useful representations during training on a new task. Freezing lower layers, keeping BatchNorm frozen, and using a small learning rate reduce this risk.

## I. Evaluation and Metrics

Let:

- `TP`: true positives
- `TN`: true negatives
- `FP`: false positives
- `FN`: false negatives

### Accuracy

Formula:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Interpretation: proportion of all correct predictions.

Strength: simple and intuitive.

Weakness: misleading under class imbalance.

Relevance: useful as a general summary, but not sufficient alone.

### Precision

Formula:

```text
Precision = TP / (TP + FP)
```

Interpretation: of all images predicted as pedestrian path, how many truly contain one?

Strength: important when false positives are costly.

Weakness: can be high even if many true paths are missed.

Relevance: measures reliability of positive predictions.

### Recall

Formula:

```text
Recall = TP / (TP + FN)
```

Interpretation: of all true pedestrian paths, how many did the model find?

Strength: important when missing positives is costly.

Weakness: can be high while producing many false positives.

Relevance: measures how well the model detects existing pedestrian paths.

### F1-Score

Formula:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Interpretation: harmonic mean of precision and recall.

Strength: balances precision and recall.

Weakness: does not include true negatives directly.

Relevance: useful for imbalanced binary classification.

### ROC-AUC

ROC-AUC measures how well the model ranks positives above negatives across thresholds. A value of `0.5` is random ranking; `1.0` is perfect ranking.

Strength: threshold-independent ranking measure.

Weakness: can appear optimistic under strong class imbalance.

Relevance: evaluates whether the predicted probabilities separate the two classes.

### Confusion Matrix

The confusion matrix shows:

```text
              predicted negative    predicted positive
true negative        TN                    FP
true positive        FN                    TP
```

It reveals the exact error types, not just aggregate scores.

### False Positives

A false positive means the model predicts pedestrian path present, but the label is negative. In this project, that could overestimate path availability.

### False Negatives

A false negative means the model predicts no pedestrian path, but the label is positive. In this project, that means missing an existing path.

### Threshold Analysis

Changing the threshold changes the precision-recall tradeoff. Increasing the threshold usually increases precision but lowers recall. Lowering the threshold usually increases recall but may produce more false positives.

## J. Error Analysis

Error analysis inspects actual mistakes instead of only looking at metrics. The project exports false-positive and false-negative CSV files, image thumbnails, and grids.

This helps identify:

- lighting bias
- perspective bias
- urban vs rural bias
- repeated scene bias
- ambiguous labels
- near-duplicate regions

Dataset bias matters because a model can perform well on familiar scenes and fail on new environments.

## K. Reproducibility

Reproducibility means that another run should produce the same data split, comparable training behavior, and the same documented artifacts under the same environment.

The project supports reproducibility through:

- fixed random seeds
- deterministic split creation
- deterministic validation/test pipelines
- saved split manifest
- saved TensorFlow/Keras versions
- saved hyperparameters
- saved split counts
- saved training histories

Reproducibility is essential in machine learning because results can otherwise be caused by randomness rather than methodology.

## L. Inference on New Data

`07_inference.py` loads the final best model and predicts on a new folder of images. It applies the same core preprocessing:

- RGB loading
- `224 x 224` target size
- bilinear interpolation
- `0-255` pixel range

It saves:

- all predictions
- positive predictions only
- copied positive images
- a static positive prediction grid

The threshold defaults to `0.5` but can be changed.

## M. Final Reflection

### Strengths

- complete reproducible workflow
- clear train/validation/test separation
- transfer learning with pretrained EfficientNetB0
- class imbalance handling
- phase-1 training and fine-tuning separation
- extensive static reporting
- threshold and error analysis
- inference support for new datasets

### Limitations

- classification only says whether a path is present, not where it is
- grouped splitting is not implemented unless group metadata becomes available
- model performance depends on dataset diversity and label quality
- test evaluation should be run only once after all decisions are fixed

### Future Improvements

- semantic segmentation to locate pedestrian paths pixel-wise
- hard negative mining to add difficult negative examples
- larger and more diverse datasets
- grouped split based on location or tile identifiers
- model comparison with ResNet, MobileNet, ConvNeXt, or EfficientNetV2
- threshold calibration based on real deployment priorities
- probability calibration using validation data

