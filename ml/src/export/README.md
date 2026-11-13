# FARZ — Model export

Export the trained `best_model.pth` to an Android-compatible ExecuTorch PTE using XNNPACK. This step does **not** retrain the model.

## Paired host/runtime versions

The Android app uses `org.pytorch:executorch-android:1.5.1`. The export notebook installs this host stack:

- `torch==2.14.0+cpu`
- `torchvision==0.29.0+cpu`
- `executorch==1.5.1`

Install from the PyTorch CPU index plus PyPI. The PyTorch 2.14.0 release pairs with TorchVision 0.29.0. ExecuTorch's 1.5 documentation uses the `torch.export` → XNNPACK lowering path.

## Install locally

~~~bash
python -m pip install --upgrade \
  torch==2.14.0+cpu torchvision==0.29.0+cpu executorch==1.5.1 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
~~~

## Export

After making the exact Experiment 01 checkpoint, manifest and images available:

~~~bash
python ml/src/export/export_executorch.py \
  --root . \
  --checkpoint ml/outputs/mobilenet_v3_small/best_model.pth \
  --manifest data/manifests/data_split.csv \
  --images-dir data/raw/dataset-resized \
  --output ml/outputs/mobilenet_v3_small/farz_model.pte \
  --parity-images 30 \
  --copy-to-app
~~~

The exporter loads the checkpoint, captures a fixed `1×3×224×224` graph, lowers to XNNPACK, serializes `.pte`, loads it in the ExecuTorch Python runtime, and checks a balanced 30-image test sample. It stops if any top-1 prediction differs. Python parity is not a substitute for a physical Android-device test.

## Input contract

- One RGB waste item per image.
- Centered square padding, RGB fill `(245, 247, 242)`.
- Bilinear resize to `224×224`.
- Float32 pixels in `[0,1]`, NCHW layout.
- ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.

Softmax scores are not calibrated probabilities.
