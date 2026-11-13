# Experiment 01 — MobileNetV3-Small

**Status:** First full Colab training run completed.  
**Runtime:** CUDA GPU  
**Initialization:** ImageNet pretrained weights  
**Best checkpoint:** epoch 12 of 17  
**Checkpoint size:** 6,227,077 bytes (~5.94 MiB)  
**Input:** RGB 224 × 224  
**Classes:** Glass, Paper, Cardboard, Plastic, Metal

## Training and validation

| Metric | Best checkpoint |
|---|---:|
| Best epoch (selected by validation Macro F1) | 12 |
| Best validation Macro F1 | 93.82% |
| Validation accuracy | 93.81% |
| Training epochs completed | 17 |
| Test set used for checkpoint selection | No |

Training accuracy approached 99.8% while validation accuracy was about 93%. This indicates a train/validation gap and some overfitting, but the validation curve remained strong. The selected checkpoint is epoch 12, not the final epoch.

## Held-out test results

| Metric | Result |
|---|---:|
| Test images evaluated | 340 |
| Correct predictions | 317 |
| Incorrect predictions | 23 |
| Accuracy | 93.24% |
| Macro Precision | 93.26% |
| Macro Recall | 93.46% |
| Macro F1 | 93.22% |
| Test loss | 0.2187 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glass | 94.37% | 91.78% | 93.06% | 73 |
| Paper | 97.56% | 94.12% | 95.81% | 85 |
| Cardboard | 98.18% | 98.18% | 98.18% | 55 |
| Plastic | 95.31% | 88.41% | 91.73% | 69 |
| Metal | 80.88% | 94.83% | 87.30% | 58 |

### Confusion matrix

Rows are true classes; columns are predicted classes. Order: Glass, Paper, Cardboard, Plastic, Metal.

| True \ Predicted | Glass | Paper | Cardboard | Plastic | Metal |
|---|---:|---:|---:|---:|---:|
| Glass | 67 | 1 | 0 | 2 | 3 |
| Paper | 1 | 80 | 1 | 0 | 3 |
| Cardboard | 0 | 1 | 54 | 0 | 0 |
| Plastic | 1 | 0 | 0 | 61 | 7 |
| Metal | 2 | 0 | 0 | 1 | 55 |

## Interpretation

- Cardboard is the strongest class in this test run (F1 98.18%).
- Metal has high recall (94.83%) but lower precision (80.88%): 13 non-metal samples were predicted as Metal. Seven of those came from Plastic.
- Plastic recall is 88.41%; the largest confusion is Plastic → Metal (7 samples).
- Paper and Cardboard remain distinct classes and both performed strongly in this test set.

Softmax top-1 scores in the prediction file are not calibrated probabilities. Do not describe them as calibrated confidence without a separate calibration step.

## Reproducibility note — test count mismatch

The submitted results bundle contains 340 test predictions. The previously committed split summary reported 341 test images. The exact `data_split.csv` used in the Colab runtime was not included in the uploaded bundle, so the one-image difference cannot yet be reconciled. The metrics above faithfully report the 340 images actually evaluated, but the split manifest from the exact Colab run must be saved with the experiment before final academic submission.

The original `test_predictions.csv` did not include source image paths. The evaluator has now been updated to include `relative_path` in future reports. No retraining is needed to use this updated evaluator, provided the same checkpoint, image set, and split manifest are retained.

## Limitations

This is a result on the held-out subset of TrashNet's mostly controlled-background images. It does not establish performance on cluttered scenes, multiple waste objects, or all real-world packaging. The next validation step is on-device export and inference parity, not another training run by default.
