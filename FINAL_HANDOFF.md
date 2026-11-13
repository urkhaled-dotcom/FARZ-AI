# FARZ — Final handoff (13 November 2026)

## Completed and verified

- Five independent classes: Glass, Paper, Cardboard, Plastic, Metal. Trash is excluded.
- Experiment 01 trained MobileNetV3-Small with PyTorch. Best checkpoint: epoch 12/17; validation Macro F1: 93.82%.
- The uploaded checkpoint was independently re-evaluated against the exact Colab manifest and original images: 340 test photos, 317 correct, test accuracy 93.2353%, Macro F1 93.2153%.
- All 2,384 image paths and SHA-256 hashes matched.
- Exact split: 1,705 train / 339 validation / 340 test. Manifest SHA-256: `9afb5036e95058134d5491e51808cb5aa952762e55f7848f10203bdba229db34`.
- Local Python test suite: 7 passed.
- `torch.export` smoke test passed; eager and exported logits matched exactly.
- Flutter app source includes camera/gallery selection, Arabic/English, Light/Dark/System themes, local preferences/history, five-class result scores and category guidance.
- Kotlin MethodChannel and ExecuTorch 1.5.1/XNNPACK Android bridge compile successfully.
- GitHub Actions run [43357785463](https://github.com/urkhaled-dotcom/FARZ-AI/actions/runs/43357785463) passed Flutter analysis, all three widget tests, and Android release APK build/upload.
- APK artifact downloaded and verified. It includes the ExecuTorch native runtime for arm64-v8a and x86_64.

## Remaining model-deployment gate: create and validate the `.pte`

The Android APK currently contains the app UI and ExecuTorch runtime/bridge, but **does not contain `farz_model.pte` yet**. I inspected the APK and found no PTE asset. The app therefore correctly displays “model unavailable” and disables classification until export parity succeeds. I am not claiming the APK currently classifies images.

The current work environment does not have ExecuTorch installed and cannot access the package index, so I could not produce the PTE here. The export notebook is prepared with the compatible CPU stack and exact Experiment 01 split.

### Export without retraining

1. Open `ml/notebooks/FARZ_Export_Colab.ipynb` in Google Colab.
2. Upload just **`FARZ-Export-Input-v1.zip`** when prompted. It contains the trained checkpoint, exact split and exporter source; no separate results ZIP or CSV upload is needed.
3. Run the notebook from top to bottom. It downloads TrashNet directly into Colab, verifies image hashes and the exact split, installs PyTorch 2.14.0+cpu / TorchVision 0.29.0+cpu / ExecuTorch 1.5.1, exports with XNNPACK and checks 30 balanced test photos.
4. Continue only when the report shows **30/30 top-1 prediction matches**. The notebook creates `FARZ-ExecuTorch-Export.zip` and saves the files to Drive.
5. Add these files to GitHub:
   - `farz_model.pte` → `app/assets/models/farz_model.pte`
   - `farz_model_metadata.json` → `app/assets/models/farz_model_metadata.json`
6. Push the assets. GitHub Actions will rebuild the APK with the real model asset. Install it on a physical Android phone and test all five classes with camera/gallery and airplane mode.

## Deliverables

- `FARZ-Android-v0.2.0-bridge.apk` — successful Android build; PTE model is not bundled yet.
- `FARZ-Project-Bundle-v0.7.zip` — project source, checkpoint, exact split, experiment reports, Colab notebooks and mobile bridge.
- `FARZ-Export-Input-v1.zip` — one-file export input with checkpoint, exact split and exporter code.
- `FARZ_Export_Colab.ipynb` — standalone export notebook.

## Limitations

- Do not report on-device classification as complete until the PTE parity check passes and the model is bundled in a rebuilt APK.
- CI success is not physical-device testing.
- FARZ v1 handles one visible waste item per image. TrashNet's controlled backgrounds do not establish performance on cluttered scenes or multiple items.
- Softmax scores are not calibrated probabilities.
