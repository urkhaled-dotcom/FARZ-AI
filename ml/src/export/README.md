# FARZ — Model export

Export the trained `best_model.pth` to an Android-compatible ExecuTorch PTE using the XNNPACK CPU backend. This step does **not** retrain the model.

## Paired versions

The Android app uses `org.pytorch:executorch-android:1.5.1`. The export environment is pinned to the same ExecuTorch release and the compatible host packages:

- Python 3.10–3.14
- `torch==2.14.0+cpu`
- `torchvision==0.29.0+cpu`
- `executorch==1.5.1`

Use the PyTorch CPU wheel index for all three packages to avoid mixing CUDA and CPU builds. The XNNPACK backend is the mobile CPU target. FARZ's training checkpoint was produced with PyTorch 2.10; its state dictionary is loaded into the compatible export environment without retraining.

## Install

In Colab, use the included `ml/notebooks/FARZ_Export_Colab.ipynb`. For a local Linux environment:

```bash
python -m pip install --upgrade \
  torch==2.14.0+cpu torchvision==0.29.0+cpu executorch==1.5.1 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

## Export command

After making the exact Experiment 01 checkpoint, split manifest, and five-class image folders available:

```bash
python ml/src/export/export_executorch.py \
  --root . \
  --checkpoint ml/outputs/mobilenet_v3_small/best_model.pth \
  --manifest data/manifests/data_split.csv \
  --images-dir data/raw/dataset-resized \
  --output ml/outputs/mobilenet_v3_small/farz_model.pte \
  --parity-images 30 \
  --copy-to-app
```

The exporter:
1. loads the trained checkpoint and validates class order;
2. captures a fixed `1×3×224×224` graph with `torch.export`;
3. lowers the graph to XNNPACK and serializes a `.pte` file;
4. loads that PTE with the ExecuTorch Python runtime;
5. compares top-1 predictions on a balanced 30-image test sample;
6. writes an export report and copies the PTE and metadata to `app/assets/models/`.

The export stops if any selected top-1 prediction differs. Python parity is not a substitute for a real Android-device test.

## Input contract

- RGB single-item image.
- Centered square padding with RGB (245, 247, 242).
- Bilinear resize to 224×224.
- Convert pixels to float32 in [0,1], NCHW layout.
- ImageNet normalization: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).

Softmax scores are not calibrated probabilities.
