# Exam Preparation Guide

## How to Present the Project in One Minute

This project solves a binary image classification problem: detecting whether an image contains a pedestrian path. The pipeline creates a reproducible stratified train/validation/test split, preprocesses all images to `224 x 224`, trains an EfficientNetB0 transfer-learning model, fine-tunes the upper backbone layers, evaluates the final model once on the test set, and exports static PNG visualizations plus CSV/JSON logs. The main methodological choices are transfer learning, Keras Functional API, class weights for imbalance, moderate augmentation, validation-loss model selection, and final threshold/error analysis.

## Likely Professor Questions and Strong Answers

### Why did you use deep learning instead of classical image processing?

Pedestrian paths vary in texture, color, lighting, perspective, and environment. Classical rules such as color thresholds or edge detectors would be brittle. A CNN learns hierarchical visual features from data and can combine low-level and high-level cues. Transfer learning further reduces the amount of data needed.

### Why EfficientNetB0?

EfficientNetB0 is a strong compromise between accuracy and computational cost. It is smaller than many older CNNs but still has good representational power. It is available in Keras with ImageNet weights, making it practical for transfer learning in a university project.

### Why not train a CNN from scratch?

Training from scratch requires more data and compute. With a limited project dataset, a scratch model would likely overfit. ImageNet pretraining gives the network useful general visual features before training starts.

### What does `include_top=False` mean?

It removes the original ImageNet classification head. The ImageNet head predicts 1000 classes, which is not useful for this binary task. We keep the convolutional feature extractor and add our own binary head.

### Why freeze the backbone in phase 1?

The new classifier head starts with random weights. If the whole network were trainable immediately, unstable gradients could damage pretrained features. Freezing the backbone lets the head first learn a stable mapping from pretrained features to the target labels.

### Why fine-tune later?

After the head has learned the task, the upper backbone layers can be carefully adapted to pedestrian-path-specific visual patterns. Lower layers remain frozen because they learn generic features.

### Why use a lower learning rate for fine-tuning?

Pretrained weights should be changed carefully. A high learning rate can destroy useful representations, a problem related to catastrophic forgetting. A small learning rate supports gradual adaptation.

### Why keep BatchNorm frozen during fine-tuning?

BatchNorm layers contain moving mean and variance statistics learned during pretraining. With a smaller target dataset, updating them can destabilize the model. Keeping BatchNorm frozen preserves stable pretrained statistics.

### Why use the Keras Functional API?

The Functional API makes the model graph explicit: input, pretrained backbone, pooling, head, output. It is more flexible than Sequential for transfer learning and fine-tuning because the backbone is a submodel and can be inspected or partially unfrozen.

### Why `224 x 224` images?

EfficientNetB0 is commonly used with `224 x 224` inputs. It is compatible with pretrained weights, computationally efficient, and still large enough to preserve scene context.

### Why no manual normalization?

Keras EfficientNetB0 includes its expected preprocessing behavior internally. Therefore the project passes images in the expected `0-255` range and avoids duplicate or inconsistent normalization.

### Why augment only the training set?

Augmentation is a training regularization technique. Validation and test data should represent real unseen data. Augmenting validation or test images would make evaluation stochastic and less interpretable.

### Why avoid aggressive augmentation?

Aggressive crops, erasing, strong blur, or large geometric distortions can remove or distort pedestrian paths. The model should learn realistic variation, not artifacts.

### Why use class weights?

The dataset has more negative than positive images. Without class weights, the model may be biased toward the majority class. Class weights increase the loss contribution of minority-class samples.

### Why use Binary Crossentropy?

Binary crossentropy is appropriate for a two-class problem with a sigmoid output. It penalizes confident wrong predictions strongly and provides differentiable gradients for optimization.

### Does the optimizer optimize accuracy?

No. Adam minimizes the loss function, binary crossentropy. Metrics such as accuracy, precision, recall, F1-score, and AUC are monitored but are not directly optimized.

### Why use EarlyStopping?

EarlyStopping monitors validation loss and stops training when it no longer improves. It helps prevent overfitting and saves computation.

### Why use ModelCheckpoint?

The best model may occur before the final epoch. ModelCheckpoint saves the model with the lowest validation loss.

### Why use ReduceLROnPlateau?

When validation loss stagnates, reducing the learning rate can help the optimizer make smaller, more precise updates.

## Metric Interpretation Questions

### Define accuracy.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

It measures the proportion of correct predictions. It is easy to understand but can be misleading when classes are imbalanced.

### Define precision.

```text
Precision = TP / (TP + FP)
```

It answers: when the model predicts pedestrian path present, how often is it correct?

### Define recall.

```text
Recall = TP / (TP + FN)
```

It answers: of all real pedestrian paths, how many did the model find?

### Define F1-score.

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

It balances precision and recall. It is useful when both false positives and false negatives matter.

### What is ROC-AUC?

ROC-AUC measures how well the model ranks positive examples above negative examples across thresholds. It is threshold-independent.

### What is a false positive here?

The model predicts pedestrian path present, but the image is labeled negative. This could overestimate pedestrian-path availability.

### What is a false negative here?

The model predicts no pedestrian path, but the image is labeled positive. This means the model missed an existing path.

### What happens if the threshold increases?

The model becomes more conservative. Precision often increases, but recall often decreases.

### What happens if the threshold decreases?

The model predicts positive more often. Recall often increases, but precision may decrease.

## Difficult Conceptual Questions

### Why can validation loss increase while training loss decreases?

The model is fitting training-specific patterns that do not generalize. This is a classic sign of overfitting.

### Why might accuracy be high but recall poor?

If the negative class is larger, predicting many negatives can produce high accuracy while missing positives. This is why recall and F1-score are important.

### Why is grouped splitting important?

If near-duplicate images or images from the same geographic area appear in both train and test, performance can be overestimated. Grouped splitting would keep related images in the same split.

### What is catastrophic forgetting?

During fine-tuning, pretrained features can be overwritten by task-specific training. A low learning rate and partial unfreezing reduce this risk.

### Why is error analysis necessary if metrics are already available?

Metrics summarize performance but do not explain failure modes. Inspecting false positives and false negatives can reveal label issues, lighting bias, environmental bias, or systematic mistakes.

## Practical Debugging Questions

### What if validation accuracy is unstable?

Check batch size, learning rate, class balance, validation split size, and whether validation data is deterministic. Also inspect whether data augmentation accidentally affects validation.

### What if training loss does not decrease?

Check labels, image loading, learning rate, model compilation, class weights, and whether the head is trainable.

### What if validation loss worsens during fine-tuning?

The learning rate may be too high, too many backbone layers may be unfrozen, or BatchNorm may be unstable. Reduce the learning rate or unfreeze fewer layers.

### What if the model predicts almost everything negative?

Possible causes include class imbalance, too high threshold, weak positive labels, insufficient positive examples, or underfitting.

### What if the model predicts too many positives?

Possible causes include too low threshold, class weights too strong, ambiguous negatives, or overfitting to positive visual patterns.

## Methodological Criticism Questions

### Why is this not semantic segmentation?

The dataset labels are image-level labels, not pixel masks. The model predicts presence or absence, not location. Semantic segmentation would be a future extension if pixel-level path masks are available.

### What are limitations of transfer learning from ImageNet?

ImageNet images may differ from aerial, tile, or domain-specific pedestrian-path imagery. Transfer learning still helps with general features, but domain mismatch can limit performance.

### Why not use a larger model?

Larger models may improve accuracy but require more compute and are more likely to overfit. EfficientNetB0 is a reasonable first model for a university project.

### Why not use Mixup or Random Erasing?

These augmentations can be useful in some tasks but may create unrealistic pedestrian-path semantics or remove the relevant path signal.

### How would you improve the project next?

Implement grouped splitting, add more diverse data, compare backbones, calibrate thresholds, use hard negative mining, and consider semantic segmentation.

## What Would Happen If...

### What if you used the test set during tuning?

The test result would no longer be an unbiased final estimate. This is data leakage.

### What if you removed class weights?

The model might become biased toward the majority negative class, possibly reducing recall for pedestrian paths.

### What if you used a threshold of 0.9?

Only very confident predictions would be positive. Precision may increase, but recall would likely decrease.

### What if you unfroze the entire backbone immediately?

Training could become unstable and pretrained features could be overwritten, especially with a limited dataset.

### What if you used no augmentation?

The model may overfit more easily and generalize worse to new lighting, scale, and perspective conditions.

