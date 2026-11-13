# FARZ — Final handoff status (13 November 2026)

## Completed and verified

- Five independent classes: Glass, Paper, Cardboard, Plastic, Metal. Trash is excluded.
- MobileNetV3-Small trained with PyTorch; best checkpoint epoch 12.
- Experiment 01 split: 2,384 images; train 1,705 / validation 339 / test 340.
- Held-out test accuracy: 93.2353% (317/340); Macro-F1: 93.2153%.
- The user-provided checkpoint loads strictly in the local PyTorch model factory and produces a finite five-logit output.
- Arabic/English Flutter UI, Light/Dark/System themes, camera/gallery, local history and launcher branding are implemented.

## Two separate Android problems identified

### 1. Galaxy A11 refuses to install the old APK

The earlier APK contained only ExecuTorch native libraries for `arm64-v8a` and `x86_64`; it did not contain `armeabi-v7a`. The Galaxy A11 family uses the Snapdragon 450 and Android 10; device ABI reports for SM-A115F show ARM32 support. The former APK was therefore not a safe package for that device. A second possible blocker for installing a later build over an older test build is an Android signing-key mismatch; uninstall the previous FARZ test build once before installing the final delivery APK.

### 2. Galaxy S22+ exits when classification begins

The app opens, but both camera and gallery lead to the same classifier method. That makes the shared native inference/preprocessing path the common failure point, rather than either image source by itself. The former path used ExecuTorch PTE/XNNPACK and could terminate in native code before Flutter could show its caught exception. Large image bitmaps also created avoidable memory pressure. Without a device logcat crash trace, the exact native fault in that old build cannot be proven; the replacement addresses both risk areas.

## Changes committed to the repository

- Replaced the Android PTE/ExecuTorch bridge with Microsoft ONNX Runtime Android 1.31.1 on CPU.
- Prepared one universal APK for `armeabi-v7a`, `arm64-v8a` and `x86_64`.
- Set Android minimum SDK to 24, which includes Android 10.
- Reduced picker resolution to 1280px and preview decode to 1024px; native decoder also downsamples before square padding.
- Limited inference threading to 2 intra-op / 1 inter-op threads.
- Disabled ONNX Runtime telemetry and removed INTERNET / ACCESS_NETWORK_STATE permissions from the merged app manifest.
- Added an Android instrumentation test which loads the bundled model and classifies a synthetic image, checking five finite softmax scores.
- CI verifies the ONNX model and C/JNI native libraries for all three ABIs, runs the inference test on an Android 10 emulator, then starts the release APK.

## Remaining mandatory model conversion

The uploaded `FARZ-Export-Input-v1.zip` was inspected. It contains the original `best_model.pth` and the exact Experiment 01 manifest; its SHA-256 matches the expected manifest. The checkpoint loads correctly. It does not contain an ONNX model yet.

Use the attached `FARZ_Universal_ONNX_Export.ipynb` in CPU-only Google Colab and upload the same `FARZ-Export-Input-v1.zip`. The notebook downloads the 2,384 source photos inside Colab, verifies all hashes, exports ONNX opset 17 and checks a balanced 30-image PyTorch-vs-ONNX Runtime parity gate. No retraining is needed.

After the notebook reports **30/30** matches, add these two files to `app/assets/models/` in GitHub:
- `farz_model.onnx`
- `farz_model_metadata.json`

That upload triggers the universal Android build. Do not use the old APK for delivery.

## Final acceptance tests

1. GitHub Actions must pass Flutter analysis, widget tests, ABI library checks, Android 10 instrumentation inference and release startup.
2. Install the same APK on the physical Galaxy A11 (Android 10).
3. Run camera and gallery classification for Glass, Paper, Cardboard, Plastic and Metal.
4. Repeat with airplane mode enabled.
5. Repeat the camera/gallery classification tests on the Galaxy S22+.
6. Confirm the app remains open and reports a result/error rather than exiting.

Python parity and an emulator test are necessary but do not replace the final physical-phone tests.
