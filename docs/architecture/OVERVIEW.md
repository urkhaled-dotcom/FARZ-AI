# FARZ — Architecture Overview

## Target scope
Course capstone, not a production-scale platform. Build an Android-first, bilingual (Arabic/English), offline-first image classifier for five classes: Glass, Paper, Cardboard, Plastic, Metal.

## Runtime flow
1. User captures or selects one image.
2. App resizes and normalizes it exactly as defined by the training pipeline.
3. On-device runtime executes the exported model.
4. App maps the five output scores to class labels.
5. App shows the predicted class, a calibrated/validated confidence indicator, and short sorting guidance.
6. Optional history and theme/language preferences stay on-device.

## Main components
- `ml/`: PyTorch data preparation, training, evaluation, and export.
- `app/`: Flutter UI and on-device inference integration.
- `data/manifests/`: dataset manifests; raw images are not committed to Git.
- `reports/`: analysis and experiment reports.

## Model deployment
PyTorch is the source of truth for training. ExecuTorch is the planned on-device runtime. Prefer its documented Android AAR + Java/Kotlin API, called from Flutter through a small MethodChannel bridge. Validate one real model on a physical Android device before the full UI is built. Do not assume the exported model is mobile-ready until it is run on a real Android device and checked against PyTorch outputs.

## Out of scope for this capstone
Cloud backend, user accounts, online inference, object detection, multi-item scene analysis, and iOS release.
