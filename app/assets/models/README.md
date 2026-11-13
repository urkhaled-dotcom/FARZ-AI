# FARZ model assets

The Android app requires the ONNX model file `farz_model.onnx` and its metadata JSON.

- Architecture: MobileNetV3-Small
- Classes, in output order: Glass, Paper, Cardboard, Plastic, Metal
- Trash is excluded.
- Input: RGB, 224×224, centered square padding RGB (245,247,242), bilinear resize, ImageNet normalization.
- Runtime: Microsoft ONNX Runtime Android 1.31.1, CPU execution provider.
- Export: ONNX opset 17 from the trained PyTorch Experiment 01 checkpoint.

Do not add an ONNX asset to this directory until the exporter reports 30/30 top-1 parity against PyTorch on the balanced test sample.

The Android workflow packages one APK with ARMv7, ARM64, x86 and x86_64 native libraries. It fails rather than producing an app with a missing model.
