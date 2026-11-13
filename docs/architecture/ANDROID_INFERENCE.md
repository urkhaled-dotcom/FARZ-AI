# FARZ Android offline inference

## Runtime pairing

- Python exporter: executorch==1.5.1
- Android Maven runtime: org.pytorch:executorch-android:1.5.1
- Backend: XNNPACK CPU
- Flutter channel: farz/inference
- Android minimum SDK: 24
- Target ABIs: arm64-v8a and x86_64

The ExecuTorch Android AAR includes the Java/Kotlin API, JNI runtime, XNNPACK backend, and supports arm64-v8a and x86_64. FARZ uses the documented Maven Central AAR and Module/Tensor/EValue API.

## Inference flow

1. Flutter's image_picker selects a photo using the Android camera or gallery intent.
2. Flutter sends only the local file path to the Kotlin MethodChannel.
3. Kotlin loads the packaged PTE model and preprocesses the photo.
4. ExecuTorch executes MobileNetV3-Small using XNNPACK on the device CPU.
5. Kotlin returns the five softmax scores, top-1 class index, and inference latency.
6. Flutter shows the category, all five class scores, sorting guidance, and saves a short local history entry.

No image is uploaded to a server.

## Input preprocessing contract

Must match the validation transform used in PyTorch:

1. Decode and apply EXIF orientation.
2. Convert to RGB.
3. Center-pad to a square with RGB (245, 247, 242).
4. Bilinear resize to 224×224.
5. Convert RGB pixel values to float32 in [0,1].
6. Normalize RGB channels by ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
7. Create contiguous NCHW tensor [1,3,224,224].

## Model asset

The PTE has been generated from the exact Experiment 01 checkpoint and passed the export notebook's 30-image balanced top-1 parity test (30/30). Add:
- app/assets/models/farz_model.pte
- app/assets/models/farz_model_metadata.json

CI copies the PTE into Android native assets before APK compilation. If the PTE is absent, the app reports that the model is not ready and keeps classification disabled rather than pretending to classify.

## Release gate

1. Run ml/notebooks/FARZ_Export_Colab.ipynb using the source bundle, training results ZIP, and canonical data_split.csv.
2. Confirm exporter output shows 30/30 top-1 matches.
3. Add the PTE and metadata files to app/assets/models and push them.
4. Wait for successful GitHub Actions analysis, widget tests, and APK build.
5. Install APK on a physical Android phone.
6. Test a sample from each of the five classes, camera and gallery input, then repeat with airplane mode enabled.
7. Record latency and compare a few predictions with PyTorch.

Python runtime parity and a successful CI build are necessary, but neither replaces the final physical Android test.
