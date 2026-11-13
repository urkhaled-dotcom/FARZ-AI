# FARZ | فَرز

**صوّر. تعرّف. افرز.**

FARZ is a focused deep-learning course final project: an Android-first, bilingual app that classifies one waste item per image into five separate material classes. The target experience is fully offline on-device inference.

## Classes

| Class | Source images |
|---|---:|
| Glass | 501 |
| Paper | 594 |
| Cardboard | 403 |
| Plastic | 482 |
| Metal | 410 |
| **Total** | **2,390** |

Trash is excluded; no classes are merged.

## Experiment 01 — verified model results

- Architecture: MobileNetV3-Small, ImageNet pretrained, PyTorch.
- Best checkpoint: epoch 12 of 17.
- Best validation Macro F1: **93.82%**.
- Held-out test: 340 photos, 317 correct.
- Test Accuracy: **93.24%**.
- Test Macro F1: **93.22%**.
- Checkpoint: 6,227,077 bytes (~5.94 MiB).

Full report: [Experiment 01](reports/experiments/MOBILENET_V3_SMALL_RUN_01.md).

The checkpoint was independently re-evaluated against the exact Colab `data_split.csv` and original image archive. All 2,384 paths and hashes matched. Exact split: **1,705 train / 339 validation / 340 test**; manifest SHA-256: `9afb5036e95058134d5491e51808cb5aa952762e55f7848f10203bdba229db34`. Keep this exact manifest with the checkpoint when reproducing Experiment 01.

## Technology

- PyTorch / TorchVision — model training and evaluation.
- Flutter / Dart — Android-first UI, Arabic/English and Light/Dark/System themes.
- ExecuTorch 1.5.1 + XNNPACK — native Android CPU runtime.
- Google Colab — data preparation, training and model export.
- GitHub Actions — CI validation and Android APK builds.

## App features implemented

- Camera and gallery selection.
- Arabic and English UI, persisted language choice.
- Light, Dark and System theme modes.
- Five-class result panel, scores, sorting tips, and local history.
- Kotlin MethodChannel bridge to ExecuTorch.
- Android preprocessing matched to training: EXIF orientation, centered square padding, 224×224 bilinear resize, RGB conversion and ImageNet normalization.

## Build status — 13 November 2026

**Android CI passed.**

GitHub Actions run [43357785463](https://github.com/urkhaled-dotcom/FARZ-AI/actions/runs/43357785463) passed Flutter analysis, all 3 widget tests, and Android release APK build/upload. The APK contains the Flutter UI and ExecuTorch native runtime/bridge for arm64-v8a and x86_64.

**The trained `.pte` model asset is not included in this APK yet.** The app intentionally displays the model-not-ready state and disables classification until export parity succeeds and `app/assets/models/farz_model.pte` is added. This is a deliberate safety check, not a pretend inference implementation.

## Export the trained model (no retraining)

Open [FARZ ExecuTorch Export Colab](ml/notebooks/FARZ_Export_Colab.ipynb). It uses the existing Experiment 01 checkpoint and exact split, downloads the images directly to Colab, verifies SHA-256 hashes, exports with ExecuTorch 1.5.1/XNNPACK, and checks top-1 parity on a balanced 30-image test sample.

Export environment: PyTorch 2.14.0+cpu, TorchVision 0.29.0+cpu, ExecuTorch 1.5.1. Install instructions are in the [export guide](ml/src/export/README.md). The notebook stops if any top-1 prediction differs.

After a successful 30/30 parity report, add:
- `farz_model.pte` → `app/assets/models/farz_model.pte`
- `farz_model_metadata.json` → `app/assets/models/farz_model_metadata.json`

Then push the files. GitHub Actions will rebuild the APK with the model asset. Install that APK on a physical Android phone and test all five classes with airplane mode enabled.

## Training notebook

[FARZ Training Colab](ml/notebooks/FARZ_Training_Colab.ipynb) documents the PyTorch training pipeline. Training has already completed; use the export notebook for mobile deployment, not another training run.

## Data integrity

- 2,390 valid source photos across the five approved classes.
- Three byte-identical cross-label duplicate pairs (six photos) are held out pending human label adjudication.
- The 2,384 Experiment 01 images all matched source SHA-256 hashes.
- Original photos were not modified or deleted; raw dataset images are not committed to GitHub.
- See [split reconciliation](reports/data_audit/SPLIT_RECONCILIATION.md) and [dataset attribution](docs/DATASET_ATTRIBUTION.md).

## Repository layout

~~~text
app/                  Flutter app and Android Kotlin inference bridge
ml/src/               PyTorch data, model, training, evaluation and export
ml/notebooks/         Colab training and ExecuTorch export notebooks
data/manifests/       Exact Experiment 01 and generated split manifests
reports/              Data audit and experiment results
docs/                 Scope, architecture, decisions, attribution and branding
scripts/              Android bridge setup for CI
.github/workflows/    Android APK build
~~~

FARZ v1 handles one visible waste item per image. TrashNet's controlled backgrounds do not establish performance on cluttered scenes or multiple items.
