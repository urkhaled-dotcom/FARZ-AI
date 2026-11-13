# FARZ — Confirmed Decisions

1. Project name: FARZ | فَرز.
2. Scope: final project for a deep learning course; keep it focused and deliverable.
3. Five independent classes: Glass, Paper, Cardboard, Plastic, Metal.
4. Trash is excluded. No class merging.
5. Current image counts: Glass 501, Paper 594, Cardboard 403, Plastic 482, Metal 410; total 2,390.
6. Original images must not be overwritten or deleted during audits.
7. Use the exact leakage-aware Experiment 01 split: 1,705 train / 339 validation / 340 test (2,384 images); six members of three cross-label duplicate pairs are held out.
8. App: Flutter, Android first, Arabic and English.
9. App appearance: Minimal & Modern, Light/Dark/System theme modes.
10. Classification must work fully offline on-device.
11. PyTorch is used for training/evaluation; ONNX opset 17 and Microsoft ONNX Runtime Android CPU are used for mobile inference.
12. Google Colab for data verification and CPU model export; GitHub for source control and Android build automation.
13. Build a single universal APK covering Android ARMv7 (32-bit), ARM64 and x86_64 emulator devices, with Android 7.0 as minimum SDK.
