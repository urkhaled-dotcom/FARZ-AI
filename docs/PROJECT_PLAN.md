# FARZ — Short Delivery Plan

This plan is intentionally limited to a course capstone.

1. **Data audit:** review 3 exact cross-label duplicate pairs and the perceptual candidates; preserve originals.
2. **Data split:** create train/validation/test by group so duplicate/near-duplicate images cannot cross splits.
3. **Model:** train MobileNetV3-Small as the mobile-oriented model; use ResNet18 as a simple comparison baseline if time allows.
4. **Evaluation:** report accuracy, macro-F1, per-class precision/recall, confusion matrix, model size, and inference timing.
5. **On-device proof-of-concept:** export the trained PyTorch model to ExecuTorch and verify predictions against PyTorch on Android.
6. **App integration:** Flutter camera/gallery, preprocessing, classification result, five-category guide, local history, Arabic/English, Light/Dark/System.
7. **Release:** automated APK build and real-phone offline test.

No cloud backend, user accounts, multi-object detection, or iOS release in this course project.
