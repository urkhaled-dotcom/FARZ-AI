# FARZ | فَرز

**صوّر. تعرّف. افرز.**

FARZ is a focused deep-learning course capstone: an Android-first, bilingual app that classifies one waste item per image into five independent material classes and is designed for fully offline on-device inference.

## Classes

| Class | Source images |
|---|---:|
| Glass | 501 |
| Paper | 594 |
| Cardboard | 403 |
| Plastic | 482 |
| Metal | 410 |
| **Total** | **2,390** |

`Trash` is excluded. No classes are merged.

## Technology

- **PyTorch / Torchvision** — training and evaluation.
- **Flutter / Dart** — Android-first UI, Arabic/English, Light/Dark/System themes.
- **ExecuTorch** — planned on-device runtime; Android integration must be validated with the trained model before release.
- **Google Colab** — data preparation and GPU training.
- **GitHub Actions** — automated Android APK build.

## Current status

- [x] Five-class scope and product direction agreed.
- [x] Visual EDA and initial data integrity audit completed.
- [x] FARZ logo SVGs, app icon concept, and brand guide created.
- [x] Flutter starter screen with Arabic/English and Light/Dark/System themes.
- [x] Android APK v0.1.0 successfully built by GitHub Actions (UI starter only; no classifier yet).
- [x] Three byte-identical image pairs with conflicting labels documented; all six files held out pending manual review.
- [x] Reproducible group-aware split generated: 1,702 train / 341 validation / 341 test.
- [x] PyTorch Dataset, image transforms, MobileNetV3-Small factory, training loop, and test evaluator implemented.
- [x] Colab notebook prepared to download the dataset directly to Colab, train, evaluate, and save results to Drive.
- [x] Local verification: seven Python tests pass; a tiny CPU smoke test also verified training, checkpoint saving, and test-report generation. These are code checks, not model-quality results.
- [x] Full Colab MobileNetV3-Small run completed: best validation Macro F1 93.82%; held-out test accuracy 93.24% and Macro F1 93.22% (340 test images). See [Experiment 01](reports/experiments/MOBILENET_V3_SMALL_RUN_01.md).
- [ ] Export the trained checkpoint to ExecuTorch and verify on Android. The checkpoint is approximately 5.94 MiB.
- [ ] Integrate camera/gallery and offline inference into Flutter.
- [ ] Final offline test on a physical Android phone.

## Data integrity

The original archive and photos are not modified or deleted. Six files from three byte-identical cross-label pairs are held out of the provisional candidate pool until human label review. This leaves 2,384 training candidates. Twenty-four additional non-identical perceptual-hash candidate pairs are grouped before splitting; they are candidates, not confirmed duplicates.

The raw dataset is intentionally not committed to GitHub. The notebook obtains the resized archive directly in Colab. See [dataset attribution](docs/DATASET_ATTRIBUTION.md).

## Start model training in Colab

1. Download the small **FARZ-ML-Training-Source.zip** bundle attached to this project conversation.
2. Open [FARZ_Training_Colab.ipynb](ml/notebooks/FARZ_Training_Colab.ipynb) in Colab.
3. Choose a GPU runtime.
4. Run the notebook from top to bottom. It downloads the 42.8 MB resized dataset into Colab, checks and splits the five classes, trains MobileNetV3-Small, evaluates on the held-out test set, and saves model/results to Google Drive.

The Colab notebook uses this direct source archive: [Gary Thung's TrashNet resized dataset](https://huggingface.co/datasets/garythung/trashnet).

## Reproduce locally

After extracting `dataset-resized.zip` into `data/raw/dataset-resized/`:

```bash
pip install -r ml/requirements.txt
python ml/src/data/audit_dataset.py --images-dir data/raw/dataset-resized --root .
python ml/src/data/build_training_candidates.py --root .
python ml/src/data/create_data_split.py --root . --seed 42
python -m pytest -q ml/tests
python ml/src/training/train.py --root . --epochs 20 --batch-size 32 --patience 5
python ml/src/evaluation/evaluate.py --root .
```

ImageNet pretrained weights are downloaded on the first training run. The test set is not used for checkpoint selection. Training outputs are saved under `ml/outputs/mobilenet_v3_small/` and should not be committed to Git.

## Repository layout

```text
app/                  Flutter Android-first app
ml/src/               PyTorch data, model, training, evaluation
ml/notebooks/         Colab training notebook
data/manifests/       Generated split manifests (not committed)
reports/data_audit/   Data quality findings
docs/                 Scope, architecture, decisions, attribution, branding
.github/workflows/    Android APK build
```

FARZ's first app release handles one waste item per image. TrashNet's simple backgrounds limit how much we can infer about cluttered real-world performance.
