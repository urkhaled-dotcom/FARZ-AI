# FARZ — Android model export

Export the trained MobileNetV3-Small checkpoint to ONNX for Microsoft ONNX Runtime on Android. This does **not** retrain the model.

## Runtime pairing

- Training checkpoint: PyTorch state dict from Experiment 01
- Export format: ONNX opset 17
- Android runtime: `com.microsoft.onnxruntime:onnxruntime-android:1.31.1`
- Execution provider: CPU
- Colab verification: `onnxruntime==1.31.1`

## Export in Google Colab

Open the attached `FARZ_Universal_ONNX_Export.ipynb` notebook delivered with this project. Upload the single `FARZ-Export-Input-v1.zip` archive when prompted. The notebook:

1. Loads the exact saved checkpoint and the original 2,384-row manifest.
2. Downloads TrashNet images directly into Colab and verifies image SHA-256 hashes.
3. Uses CPU-only PyTorch, TorchVision, ONNX and ONNX Runtime.
4. Exports `farz_model.onnx` without retraining.
5. Checks 30 balanced test images and stops unless top-1 predictions match 30/30.
6. Produces an export ZIP for the Android asset folder.

## Equivalent command

After the matching CPU environment is installed:

```bash
python ml/src/export/export_onnx.py \
  --root . \
  --checkpoint ml/outputs/mobilenet_v3_small/best_model.pth \
  --manifest data/manifests/data_split_run_01.csv \
  --images-dir data/raw/dataset-resized \
  --output ml/outputs/mobilenet_v3_small/farz_model.onnx \
  --parity-images 30 \
  --copy-to-app
```

Install dependencies compatible with the training checkpoint and ONNX export stack. The Colab notebook pins the versions and verifies them.

## Android input contract

- RGB image, one waste object per image
- Centered square padding with RGB (245, 247, 242)
- Bilinear resize to 224×224
- Float32 NCHW input
- ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)

A 30/30 Python parity report is not a physical-device test. The final release still must install and run on the Galaxy A11 and a recent Android phone.
