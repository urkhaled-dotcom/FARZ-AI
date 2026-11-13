# Experiment 01 — MobileNetV3-Small (canonical manifest verified)

**Status:** Full Colab training and held-out evaluation completed; test metrics independently reproduced from the uploaded checkpoint, exact split, and original source images.

## Configuration

- Architecture: Torchvision MobileNetV3-Small
- Initialization: ImageNet pretrained weights
- Training device: CUDA GPU
- Evaluation reproduced locally on CPU
- Input: RGB 224 × 224
- Classes/order: Glass, Paper, Cardboard, Plastic, Metal
- Best checkpoint: epoch 12 of 17, selected by validation Macro F1
- Checkpoint size: 6,227,077 bytes (~5.94 MiB)
- Original Trash class excluded; no class merging

## Canonical split used in this run

The exact Colab data_split.csv uploaded after training was verified against the original dataset-resized archive.

| Split | Images |
|---|---:|
| Train | 1,705 |
| Validation | 339 |
| Test | 340 |
| **Total** | **2,384** |

All 2,384 paths exist; every image SHA-256 matches the uploaded manifest. There are 2,360 unique perceptual groups, no group leakage, and no repeated SHA-256 values in the selected manifest.

Manifest SHA-256:

`9afb5036e95058134d5491e51808cb5aa952762e55f7848f10203bdba229db34`

The canonical manifest is preserved in the project bundle as data/manifests/data_split_run_01.csv. An earlier repository summary (1,702 / 341 / 341) described a separately regenerated split and was not used for this run. Do not replace this exact manifest when reproducing Experiment 01.

## Training / validation

| Metric | Result |
|---|---:|
| Best checkpoint epoch | 12 / 17 |
| Validation accuracy | 93.81% |
| Validation Macro F1 | 93.82% |
| Training accuracy at epoch 12 | 99.77% |

The train/validation gap indicates some overfitting. The selected checkpoint is epoch 12, not the final epoch.

## Held-out test results

| Metric | Result |
|---|---:|
| Test images | 340 |
| Correct predictions | 317 |
| Incorrect predictions | 23 |
| Accuracy | 93.24% |
| Macro Precision | 93.26% |
| Macro Recall | 93.46% |
| Macro F1 | 93.22% |
| Test loss | 0.2187 |

### Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glass | 94.37% | 91.78% | 93.06% | 73 |
| Paper | 97.56% | 94.12% | 95.81% | 85 |
| Cardboard | 98.18% | 98.18% | 98.18% | 55 |
| Plastic | 95.31% | 88.41% | 91.73% | 69 |
| Metal | 80.88% | 94.83% | 87.30% | 58 |

### Confusion matrix

Rows are true classes; columns are predicted classes. Order: Glass, Paper, Cardboard, Plastic, Metal.

| True \\ Predicted | Glass | Paper | Cardboard | Plastic | Metal |
|---|---:|---:|---:|---:|---:|
| Glass | 67 | 1 | 0 | 2 | 3 |
| Paper | 1 | 80 | 1 | 0 | 3 |
| Cardboard | 0 | 1 | 54 | 0 | 0 |
| Plastic | 1 | 0 | 0 | 61 | 7 |
| Metal | 2 | 0 | 0 | 1 | 55 |

## Interpretation

- Cardboard is strongest on this test set (F1 98.18%).
- Metal has high recall (94.83%) but lower precision (80.88%): 13 non-metal photos were predicted as Metal, including seven Plastic photos.
- Plastic recall is 88.41%; the largest confusion is Plastic → Metal (7 cases).
- Paper and Cardboard remain separate classes and both perform strongly.
- Raw softmax scores are not calibrated probabilities.

## Independent reproducibility check

The checkpoint was reloaded in PyTorch 2.10.0 CPU and evaluated using the exact uploaded split and dataset images. All 340 predictions were reproduced and the following metrics matched the Colab output exactly at displayed precision: Accuracy 93.2353%, Macro Precision 93.2608%, Macro Recall 93.4627%, Macro F1 93.2153%.

A separate torch.export graph was also created for a sample input; its eager/exported logits matched exactly (maximum absolute difference 0.0). This is not yet an ExecuTorch PTE export.

## Next deployment gate

Export this exact checkpoint to ExecuTorch 1.5.1/XNNPACK, verify PTE predictions against PyTorch on a balanced test sample, then build and test on a physical Android phone. Python parity does not replace real-device validation.

The reported metrics apply to TrashNet's controlled-background photos. They do not establish equivalent performance on cluttered scenes or images containing multiple waste items.
