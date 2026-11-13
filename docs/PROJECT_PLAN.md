# FARZ — Short Delivery Plan

This plan is intentionally limited to a final project for a deep learning course.

1. **Data audit — completed:** inspect the five approved classes, validate image files, hold out conflicting exact duplicates, and group near-duplicate candidates.
2. **Data split — completed:** create a reproducible stratified group split (1,702 train / 341 validation / 341 test).
3. **Training pipeline — implemented:** PyTorch Dataset/transforms, MobileNetV3-Small, training loop, validation checkpoint selection, mixed precision, and early stopping.
4. **Colab training — ready to run:** train on GPU using the included notebook. Save the best checkpoint and experiment log to Google Drive.
5. **Evaluation — implemented:** evaluate once on the held-out test set; report accuracy, macro-F1, per-class precision/recall, and confusion matrix.
6. **Mobile proof-of-concept:** export the trained PyTorch model with ExecuTorch and verify output agreement on Android.
7. **App integration and release:** connect Flutter camera/gallery, preprocessing, offline inference, result page, themes, and language controls; build and test the APK.

Do not add a backend, user accounts, multi-object detection, or iOS release in this course project.
