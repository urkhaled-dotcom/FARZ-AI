# FARZ — Model export

Export the trained `best_model.pth` to an Android-compatible ExecuTorch PTE using the XNNPACK CPU backend.

## Version pairing

FARZ pins the Python exporter and Android runtime to ExecuTorch **1.5.1**. The Android artifact is `org.pytorch:executorch-android:1.5.1`.

## Colab command

After installing the exporter and making the exact training checkpoint, split manifest, and five-class image folders available:

```bash
pip install executorch==1.5.1
python ml/src/export/export_executorch.py \
  --root . \
  --checkpoint ml/outputs/mobilenet_v3_small/best_model.pth \
  --manifest data/manifests/data_split.csv \
  --images-dir data/raw/dataset-resized \
  --output ml/outputs/mobilenet_v3_small/farz_model.pte \
  --parity-images 32 \
  --copy-to-app
```

The exporter:
1. loads the exact trained checkpoint and verifies class order;
2. calls `torch.export.export` with a fixed `1×3×224×224` input;
3. lowers the graph to XNNPACK and serializes a `.pte` file;
4. loads that file using the ExecuTorch Python runtime;
5. compares top-1 predictions against PyTorch on up to 32 held-out test images;
6. writes an export report and copies the PTE plus preprocessing metadata to `app/assets/models/`.

The export is rejected if the PTE runtime top-1 predictions differ on the parity sample. The parity test does not replace physical Android-device validation.

## Input contract (must match Android preprocessing)

- RGB image, one waste item.
- Pad to square using RGB (245, 247, 242), centered.
- Resize to 224×224 using bilinear interpolation.
- Convert to NCHW float32 and divide pixels by 255.
- Normalize with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).

Softmax scores displayed by the app are model scores, not calibrated probabilities.
