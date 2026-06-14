# Code Explanation: `00_baseline.py`

## File Purpose

This script computes a simple majority-class baseline. It uses only the class
counts from the training split, finds the most frequent training class, and
predicts that class for every validation and test image.

## Workflow

1. Count positive and negative images in the training split.
2. Select the majority class from training only.
3. Evaluate the same constant prediction on validation and test.
4. Save baseline metrics as JSON and CSV in `artifacts/baseline/`.

## Design Reasoning

The baseline does not use image pixels and therefore cannot solve the visual
task. Its purpose is methodological: the deep learning model must clearly beat
this non-learning reference before transfer learning is defensible.

Because the dataset contains more negative than positive samples, a
majority-class baseline can obtain non-trivial accuracy while having poor or
zero positive-class recall. This makes it useful for explaining why accuracy
alone is insufficient.

## Potential Pitfalls

- The baseline is not a competitive computer-vision model.
- It should not be used as evidence that the task is solved.
- It is a lower bound for comparison, not a replacement for stronger baselines
  such as logistic regression on frozen features or a lightweight CNN.
