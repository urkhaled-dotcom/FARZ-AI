# FARZ Android offline inference

## Runtime

- Training framework: PyTorch MobileNetV3-Small
- Deployment format: ONNX (opset 17)
- Android inference runtime: Microsoft ONNX Runtime Android 1.31.1
- Default execution provider: CPU
- Android minimum SDK: 24 (Android 7.0)
- Android package targets: armeabi-v7a, arm64-v8a, x86 and x86_64
- Flutter channel: farz/inference

The app uses one universal APK containing the same application and model for supported Android ABIs. The ONNX Runtime Android Maven package is used instead of the former ExecuTorch PTE AAR, which did not include an ARMv7 runtime.

## Inference flow

1. Flutter's image_picker selects a photo using the Android camera or gallery intent.
2. Flutter sends only the local file path to the Kotlin MethodChannel.
3. Kotlin loads `farz_model.onnx` from Android native assets.
4. ONNX Runtime executes MobileNetV3-Small on the device CPU.
5. Kotlin returns the five class scores, top-1 class index, and inference latency.
6. Flutter shows the category and sorting guidance, then saves a short local history entry.

No image is uploaded to a server. Classification is local and works without an internet connection after installation.

## Input preprocessing contract

Must match the PyTorch evaluation transform:

1. Decode and apply EXIF orientation.
2. Convert to RGB.
3. Center-pad to a square with RGB (245, 247, 242).
4. Bilinear resize to 224×224.
5. Convert RGB values to float32 in [0,1].
6. Normalize RGB channels by ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
7. Create contiguous NCHW tensor [1,3,224,224].

## Model asset

Required files:
- `app/assets/models/farz_model.onnx`
- `app/assets/models/farz_model_metadata.json`

CI copies the ONNX file into Android native assets. It refuses to build an APK if the model is absent. The export workflow must show 30/30 top-1 matches between PyTorch and ONNX Runtime on a balanced 30-image held-out sample before packaging.

## Release gate

1. Export the exact saved Experiment 01 checkpoint to ONNX without retraining.
2. Verify PyTorch vs ONNX Runtime 30/30 top-1 parity.
3. Add `farz_model.onnx` and metadata to `app/assets/models/`.
4. Run GitHub Actions: Flutter analysis, widget tests, universal APK build and ABI-content checks.
5. Install and test on the physical Samsung Galaxy A11 running Android 10.
6. Test camera and gallery, all five classes, and airplane mode.
7. Record inference latency and compare selected predictions with PyTorch.

Python ONNX Runtime parity and a successful CI build are necessary, but neither replaces testing on the physical delivery device.
