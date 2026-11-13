# FARZ — Final handoff status (13 November 2026)

## Project contents

- Five independent material classes: Glass, Paper, Cardboard, Plastic and Metal. Trash is excluded.
- PyTorch MobileNetV3-Small trained on the prepared five-class subset.
- Experiment 01: 2,384 images; train 1,705 / validation 339 / test 340.
- Reported held-out test accuracy: 93.2353% (317/340); Macro-F1: 93.2153%.
- Flutter Android application with Arabic/English UI, Light/Dark/System themes, camera/gallery, local history and offline inference architecture.
- Application display name: FARZ | فرز.

## Installation diagnosis

The prior APK was built with ExecuTorch 1.5.1. Inspection of that APK confirmed it contained only `arm64-v8a` and `x86_64` native ExecuTorch libraries, not `armeabi-v7a`. Some Samsung Galaxy A11 system builds expose the 32-bit ARM ABI only even though the Snapdragon 450 CPU itself is 64-bit. This is a likely reason the earlier 64-bit-only APK was rejected with “App not installed”. It was not valid to call that APK universally compatible.

The old emulator smoke test also had a workflow shell error before the install command ran; therefore it did not prove physical or emulator installation.

## Current compatibility migration

The Android bridge and build configuration have been migrated to Microsoft ONNX Runtime Android 1.31.1. The new build is configured for one universal APK containing:
- `armeabi-v7a`
- `arm64-v8a`
- `x86`
- `x86_64`

The startup model-availability check now looks for the ONNX asset. The Android 10 (API 29) emulator smoke test uses a POSIX-compatible shell script. The CI workflow checks for all four runtime libraries before uploading the APK.

## Remaining mandatory model conversion

The existing PTE model cannot be loaded by ONNX Runtime. The original trained checkpoint is needed to create the ONNX model. The repository currently does not yet contain `app/assets/models/farz_model.onnx`; the build script intentionally stops rather than packaging a non-working app.

Use `FARZ_Universal_ONNX_Export.ipynb` with the existing `FARZ-Export-Input-v1.zip` in CPU-only Google Colab. This conversion does not retrain the model. Continue only if the notebook reports 30/30 PyTorch-vs-ONNX Runtime top-1 matches. Upload the resulting `farz_model.onnx` and metadata to `app/assets/models/` in GitHub.

## Final acceptance tests

After the ONNX asset is committed and the new CI run succeeds:
1. Verify the APK contains the ONNX model and ARMv7/ARM64/x86/x86_64 runtime libraries.
2. Verify startup on Android 10 API 29 emulator.
3. Install on the physical Samsung Galaxy A11 (Android 10).
4. Test camera and gallery with Glass, Paper, Cardboard, Plastic and Metal examples.
5. Repeat classification with airplane mode enabled.
6. Install the same APK on a recent 64-bit Android phone.
7. If installing over an older FARZ test build reports a signature conflict, uninstall the old test build once before installing the final build. Future updates must use a stable signing key.

The reported PyTorch dataset score is not a claim of measured accuracy on the physical Android phone. The physical installation and on-device inference tests remain the final release gate.
