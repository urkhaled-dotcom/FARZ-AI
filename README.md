# FARZ | فَرز

**صوّر. تعرّف. افرز.**

FARZ is a focused final project for a deep-learning course: an Android-first, bilingual app that classifies one waste item per image into five independent material classes and is designed for fully offline on-device inference.

## Classes

| Class | Source images |
|---|---:|
| Glass | 501 |
| Paper | 594 |
| Cardboard | 403 |
| Plastic | 482 |
| Metal | 410 |
| **Total** | **2,390** |

Trash is excluded. No classes are merged.

## Current verified results — Experiment 01

- **Model:** MobileNetV3-Small, ImageNet pretrained, PyTorch.
- **Best checkpoint:** epoch 12 of 17.
- **Validation Macro F1:** 93.82%.
- **Test set:** 340 images; 317 correct.
- **Test accuracy:** 93.24%.
- **Test Macro F1:** 93.22%.
- **Checkpoint:** 6,227,077 bytes (~5.94 MiB).

See [full experiment report](reports/experiments/MOBILENET_V3_SMALL_RUN_01.md).

The test metrics were independently reproduced from the uploaded Colab checkpoint, exact split manifest, and original dataset archive. All 2,384 image paths and SHA-256 hashes matched. The exact Experiment 01 split is **1,705 train / 339 validation / 340 test**, with manifest SHA-256 **9afb5036e95058134d5491e51808cb5aa952762e55f7848f10203bdba229db34**. Keep that manifest with the checkpoint to reproduce these metrics; do not replace it with a newly generated split.

## Technology

- **PyTorch / Torchvision** — model training and evaluation.
- **Flutter / Dart** — Android-first UI, Arabic/English, Light/Dark/System themes.
- **ExecuTorch 1.5.1 + XNNPACK** — Android runtime; final PTE export and physical-device validation are release gates.
- **Google Colab** — data preparation, GPU training, and PTE export.
- **GitHub Actions** — automated Android APK build.

## App features implemented

- Camera and gallery image selection.
- Arabic / English interface.
- Light / Dark / System appearance with local preference storage.
- Five-class results view, class score bars, sorting guidance, and local history.
- Kotlin MethodChannel bridge for local ExecuTorch inference.
- Android preprocessing aligned with model evaluation: EXIF correction, centered square padding, 224×224 bilinear resize, RGB conversion, and ImageNet normalization.

**Model status:** the trained .pth checkpoint has been verified, and the Android bridge is implemented. The .pte model asset must still be exported in Colab and added at app/assets/models/farz_model.pte. Until that file is added and the app is rebuilt, the camera/gallery UI works but classification remains disabled. Do not treat mobile deployment as finished before PTE parity and real-phone tests pass.

## Data integrity

- 2,390 valid source photos across the five approved classes.
- Three byte-identical cross-label pairs (six files) are held out pending human label adjudication.
- The 2,384 images used in Experiment 01 all match their source SHA-256 hashes.
- No original image was deleted or modified.
- The raw dataset is not committed to GitHub.
- The exact Experiment 01 manifest is preserved in the downloadable project bundle as data/manifests/data_split_run_01.csv.

See [split reconciliation](reports/data_audit/SPLIT_RECONCILIATION.md) and [dataset attribution](docs/DATASET_ATTRIBUTION.md).

## Training and export notebooks

1. [FARZ Training Colab](ml/notebooks/FARZ_Training_Colab.ipynb) — reproduces the PyTorch training pipeline.
2. [FARZ ExecuTorch Export Colab](ml/notebooks/FARZ_Export_Colab.ipynb) — uses the existing checkpoint and exact Experiment 01 split, exports a PTE with XNNPACK, and checks top-1 parity across a balanced 30-image test sample.

The export notebook downloads the 42.8 MB resized TrashNet archive directly into Colab. It does not require uploading the dataset from your laptop.

### After PTE export

Copy these two files from the export package into the repository:
- farz_model.pte → app/assets/models/farz_model.pte
- farz_model_metadata.json → app/assets/models/farz_model_metadata.json

Then push the commit. GitHub Actions will prepare the Android Kotlin bridge, bundle the PTE as a native asset, run Flutter analysis and widget tests, and build a release APK.

## Local development

~~~bash
pip install -r ml/requirements.txt
python ml/src/data/audit_dataset.py --images-dir data/raw/dataset-resized --root .
python ml/src/data/build_training_candidates.py --root .
python ml/src/data/create_data_split.py --root . --seed 42
python -m pytest -q ml/tests
python ml/src/training/train.py --root . --epochs 20 --batch-size 32 --patience 5
python ml/src/evaluation/evaluate.py --root .
~~~

The command above regenerates a new split, so it will not reproduce Experiment 01. Use data_split_run_01.csv from the project bundle when reproducing the recorded metrics.

## Repository layout

~~~text
app/                  Flutter app and Android Kotlin inference bridge
ml/src/               PyTorch data, model, training, evaluation, and export
ml/notebooks/         Training and ExecuTorch export Colab notebooks
data/manifests/       Experiment 01 and generated split manifests
reports/              Data audit and experiment results
docs/                 Scope, architecture, decisions, attribution, branding
scripts/              Android bridge setup for CI
.github/workflows/    Android APK build
~~~

FARZ v1 classifies one waste item per image. TrashNet's mostly simple backgrounds do not establish performance for cluttered scenes or multiple waste items.
