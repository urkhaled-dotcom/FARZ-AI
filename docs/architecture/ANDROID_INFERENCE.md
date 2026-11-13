# FARZ Android offline inference

## Runtime

- Training framework: PyTorch MobileNetV3-Small
- Deployment format: ONNX (opset 17)
- Android inference runtime: Microsoft ONNX Runtime Android 1.31.1
- Default execution provider: CPU
- Android minimum SDK: 24 (Android 7.0)
- Android package targets: armeabi-v7a, arm64-v8a and x86_64
- Flutter channel: farz/inference

The app uses one universal APK containing the same application and model for ARMv7 Android 32-bit phones, ARM64 phones and x86_64 emulator devices. The Android AAR is verified by CI for its ONNX Runtime C and Java JNI libraries for each target ABI.

## Inference flow

1. Flutter's image_picker selects a photo using the Android camera or gallery intent.
2. Flutter bounds the selected image to a manageable resolution and sends only its local file path to the Kotlin MethodChannel.
3. Kotlin decodes the image with a bounded sample size, applies EXIF orientation and creates the model tensor.
4. ONNX Runtime executes MobileNetV3-Small locally on the device CPU.
5. Kotlin returns five class scores, the top-1 class index and inference latency.
6. Flutter shows the category and sorting guidance, then saves a short local history entry.

No image is uploaded to a server. Classification is local and works without an internet connection after installation. The manifest merge removes INTERNET and ACCESS_NETWORK_STATE permissions, and ONNX Runtime telemetry is disabled.

## Input preprocessing contract

Must match the PyTorch evaluation transform:

1. Decode and apply EXIF orientation.
2. Convert to RGB.
3. Center-pad to a square with RGB (245, 247, 242).
4. Bilinear resize to 224×224.
5. Convert RGB values to float32 in [0,1].
6. Normalize RGB channels by ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
7. Create contiguous NCHW tensor [1,3,224,224].

Camera/gallery originals are bounded to a 1,024-pixel maximum decode edge before tensor preparation to avoid unnecessary bitmap allocations on low-memory devices.

## Model asset

Required files:
- `app/assets/models/farz_model.onnx`
- `app/assets/models/farz_model_metadata.json`

CI refuses to build an APK if the ONNX file is absent. The Colab export notebook uses the saved Experiment 01 checkpoint and stops unless PyTorch and ONNX Runtime match on a balanced sample of 30 held-out images.

## Release gate

1. Export the exact saved Experiment 01 checkpoint to ONNX without retraining.
2. Verify PyTorch vs ONNX Runtime 30/30 top-1 parity.
3. Add `farz_model.onnx` and metadata to `app/assets/models/`.
4. Run GitHub Actions: Flutter analysis, widget tests, universal APK build, ABI-content checks and Android 10 instrumentation inference test.
5. Install and test on the physical Samsung Galaxy A11 running Android 10.
6. Test camera and gallery, all five classes, and airplane mode.
7. Repeat the same classification test on the Galaxy S22+ to confirm the previous inference crash is resolved.

Python ONNX Runtime parity and a successful Android emulator test are necessary, but neither replaces testing on both physical phones.
