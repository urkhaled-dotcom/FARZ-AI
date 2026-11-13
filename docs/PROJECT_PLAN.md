# FARZ — Short Delivery Plan

This plan is intentionally limited to a final project for a deep learning course.

1. **Data audit — completed:** inspect the five approved classes, validate image files, hold out conflicting exact duplicates, and group near-duplicate candidates.
2. **Data split — completed:** reproducible leakage-aware split: 1,705 train / 339 validation / 340 test (2,384 images).
3. **Training pipeline — completed:** PyTorch Dataset/transforms, MobileNetV3-Small, training loop, validation checkpoint selection, mixed precision, and early stopping.
4. **Colab training — completed:** Experiment 01 checkpoint and experiment log saved; do not retrain for mobile deployment.
5. **Evaluation — completed:** held-out test accuracy 93.2353% (317/340) and Macro-F1 93.2153%.
6. **Mobile export — prepared:** convert the supplied checkpoint to ONNX on CPU and require 30/30 PyTorch-vs-ONNX top-1 matches before release.
7. **App integration and release — implementation updated:** Flutter camera/gallery, bounded image preprocessing, Arabic/English, themes, local history, offline ONNX inference, one universal Android APK and CI inference smoke test.

Do not add a backend, user accounts, multi-object detection, or iOS release in this course project.

The Android APK is not ready for delivery until the ONNX model has been exported, passed parity and been bundled.
