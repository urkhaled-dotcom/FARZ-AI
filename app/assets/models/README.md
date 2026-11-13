# FARZ on-device model assets

The Experiment 01 MobileNetV3-Small checkpoint has been exported successfully to ExecuTorch 1.5.1 with the XNNPACK backend.

## Required files

Place both files in this directory:

- `farz_model.pte` — 6,124,436-byte on-device model.
- `farz_model_metadata.json` — class order and preprocessing contract.

Export verification:

- Classes: Glass, Paper, Cardboard, Plastic, Metal.
- Input: RGB, 224×224, centered square padding with RGB (245, 247, 242), bilinear resize, ImageNet normalization.
- ExecuTorch Python runtime parity: **30/30 top-1 matches** against PyTorch.
- Maximum absolute logit error: 0.00004268.
- PTE SHA-256: `74ebef11b76292696b9f69f588ea550c90a978c447038493f5ad5b8a02018195`.

## Android packaging

Commit both files to this folder. The GitHub Actions workflow runs on changes under `app/**`. It invokes `scripts/prepare_android.py`, which copies the PTE into Android native assets at `app/android/src/main/assets/models/farz_model.pte` before the APK is built.

Python parity has passed. It does not replace testing the built APK on a physical Android device, including classification with airplane mode enabled.
