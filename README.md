# FARZ | فَرز

**صوّر. تعرّف. افرز.**

FARZ is a focused deep-learning course final project: an Android-first, bilingual app that classifies one waste item per image into five separate material classes. The target experience is fully offline using on-device inference.

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

## Experiment 01 — PyTorch results

- Architecture: MobileNetV3-Small, ImageNet pretrained, PyTorch.
- Best checkpoint: epoch 12 of 17.
- Best validation Macro F1: **93.82%**.
- Held-out test: 340 photos, 317 correct.
- Test Accuracy: **93.24%**.
- Test Macro F1: **93.22%**.
- Checkpoint: 6,227,077 bytes (~5.94 MiB).

Full report: [Experiment 01](reports/experiments/MOBILENET_V3_SMALL_RUN_01.md).

The checkpoint was independently re-evaluated against the exact Colab manifest and original image archive. All 2,384 paths and hashes matched. Exact split: **1,705 train / 339 validation / 340 test**; manifest SHA-256: `9afb5036e95058134d5491e51808cb5aa952762e55f7848f10203bdba229db34`.

These are PyTorch test-set results, not measurements from a physical Android phone.

## Technology

- PyTorch / TorchVision — training and evaluation.
- ONNX — model exchange format for mobile deployment.
- Microsoft ONNX Runtime Android 1.31.1 — local Android CPU inference.
- Flutter / Dart — Android UI, Arabic/English and Light/Dark/System themes.
- Google Colab — data preparation, training and CPU model export.
- GitHub Actions — validation and universal Android APK builds.

## App features

- Camera and gallery selection.
- Arabic and English UI, persisted language choice.
- Light, Dark and System theme modes.
- Five-class result panel, scores, sorting tips, and local history.
- Kotlin MethodChannel bridge to ONNX Runtime.
- Preprocessing matched to training: EXIF orientation, centered square padding, 224×224 bilinear resize, RGB conversion and ImageNet normalization.
- No account or backend; the photo is processed locally.

## Android compatibility fix

The first APK used an ExecuTorch AAR containing only `arm64-v8a` and `x86_64` runtime libraries. It did not include `armeabi-v7a`. That is a likely reason it was refused by some Galaxy A11 system builds, despite Android 10 being new enough.

The current repository has migrated the Android inference bridge and build configuration to ONNX Runtime Android 1.31.1. The intended single APK includes:
- `armeabi-v7a`
- `arm64-v8a`
- `x86`
- `x86_64`

CI checks that the ONNX model and all four native runtimes are actually present, and runs an Android 10 API 29 startup smoke test. The final compatibility claim must still be confirmed on the physical Galaxy A11.

## Remaining deployment step — no retraining

The trained checkpoint is not yet converted to ONNX. The old `.pte` file cannot be loaded by ONNX Runtime, so **do not use the old APK** as the delivery version.

Open the attached **`FARZ_Universal_ONNX_Export.ipynb`** notebook delivered with this project and upload `FARZ-Export-Input-v1.zip`. The notebook downloads the dataset directly inside Colab, uses a CPU-only export environment, converts the existing checkpoint without retraining, and checks PyTorch vs ONNX Runtime parity on 30 balanced held-out test images. It stops unless all 30 predictions match.

After a successful 30/30 report:
1. Upload `farz_model.onnx` to `app/assets/models/farz_model.onnx`.
2. Upload `farz_model_metadata.json` to `app/assets/models/farz_model_metadata.json`.
3. Push the files. GitHub Actions will build and validate the universal APK.

The build intentionally refuses to create an APK when the ONNX model is missing.

## Training notebook

[FARZ Training Colab](ml/notebooks/FARZ_Training_Colab.ipynb) documents training. Training has already completed; use the ONNX export notebook for mobile deployment, not another training run.

## Data integrity

- 2,390 valid source photos across the five approved classes.
- Three byte-identical cross-label duplicate pairs (six photos) are held out pending label review.
- All 2,384 Experiment 01 images matched source SHA-256 hashes.
- Original photos were not modified or deleted; raw dataset images are not committed to GitHub.
- See [split reconciliation](reports/data_audit/SPLIT_RECONCILIATION.md) and [dataset attribution](docs/DATASET_ATTRIBUTION.md).

## Repository layout

~~~text
app/                  Flutter app and Android Kotlin inference bridge
ml/src/               PyTorch data, model, training, evaluation and export
ml/notebooks/         Colab training and ONNX export notebooks
data/manifests/       Exact Experiment 01 split
reports/              Data audit and experiment results
docs/                 Scope, architecture, decisions, attribution and branding
scripts/              Android bridge setup for CI
.github/workflows/    Universal Android APK build and checks
~~~

FARZ v1 handles one visible waste item per image. TrashNet's controlled backgrounds do not establish performance on cluttered scenes or multiple items.
