# FARZ — Architecture Overview

## Target scope
Course capstone, not a production-scale platform. Build an Android-first, bilingual (Arabic/English), offline-first image classifier for five classes: Glass, Paper, Cardboard, Plastic, Metal.

## Runtime flow
1. User captures or selects one image.
2. Flutter bounds the image preview and sends the local image path to the Android inference bridge.
3. Kotlin decodes and preprocesses the image to match the training pipeline.
4. ONNX Runtime executes the exported MobileNetV3-Small model on the CPU.
5. App maps the five output scores to class labels.
6. App shows the predicted class and sorting guidance; history and preferences remain on-device.

## Main components
- `ml/`: PyTorch data preparation, training, evaluation, and ONNX export.
- `app/`: Flutter UI and Android ONNX Runtime integration.
- `data/manifests/`: dataset manifests; raw images are not committed to Git.
- `reports/`: analysis and experiment results.

## Model deployment
PyTorch is the source of truth for training. The trained checkpoint is exported to ONNX opset 17 and validated with ONNX Runtime before inclusion in the APK. The Android app uses the Microsoft ONNX Runtime AAR and a small Kotlin MethodChannel bridge. The universal APK includes ARMv7, ARM64 and x86_64 libraries. CI runs a real synthetic-image inference test on an Android 10 emulator before accepting the release build.

## Out of scope for this capstone
Cloud backend, user accounts, online inference, object detection, multi-item scene analysis, and iOS release.
