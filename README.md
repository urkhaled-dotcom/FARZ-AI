# FARZ | فَرز

**صوّر. تعرّف. افرز.**

FARZ is a focused deep-learning course capstone: an Android-first, bilingual mobile app that classifies one waste item per image into five independent material classes and is designed to run offline on the device.

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

- **PyTorch / Torchvision** — model training and evaluation.
- **Flutter / Dart** — Android-first UI with Arabic/English and Light/Dark/System themes.
- **ExecuTorch** — planned on-device runtime; Android integration will be validated before being treated as complete.
- **Google Colab** — data preparation and model training.
- **GitHub Actions** — automated Android APK build.

## Current status

- [x] Project scope and five classes agreed.
- [x] Initial visual EDA and data integrity audit completed.
- [x] Initial FARZ logo SVGs and brand guide created.
- [x] Flutter home-screen and theme starter committed.
- [x] Arabic/English home-screen labels and Light/Dark/System theme controls added.
- [x] Exact cross-label duplicate pairs documented and held out pending manual review.
- [x] Dataset audit scripts run against the provided dataset archive: 2,390 valid images, zero unreadable images, three exact cross-label duplicate groups.
- [x] Local data-contract tests pass (2 tests).
- [ ] Review remaining near-duplicate candidates and label quality.
- [x] Create a reproducible leakage-aware split: 1,702 train / 341 validation / 341 test; near-duplicate candidates grouped before splitting.
- [ ] Train and evaluate a compact PyTorch model.
- [ ] Validate ExecuTorch export and Android runtime.
- [ ] Integrate camera/gallery and offline inference.
- [ ] Build and test Android APK on a physical device.

## Data integrity note

The original archive must remain unchanged. Three byte-identical image pairs were found under conflicting labels; all six files are excluded from the provisional training-candidate manifest until manual adjudication. This leaves a provisional candidate pool of 2,384 images. See `reports/data_audit/LABEL_REVIEW_NOTES.md`. No source image is deleted or relabelled automatically.

The source dataset is not committed to this repository. Keep the original archive in Google Drive or another private location and regenerate manifests using the audit scripts.

## Reproduce the data audit

From the repository root, after making the original ZIP available locally:

```bash
pip install -r requirements.txt
python ml/src/data/audit_dataset.py --archive /path/to/dataset-resized.zip --root .
python ml/src/data/build_training_candidates.py --root .
python ml/src/data/create_data_split.py --root .
python -m pytest -q
```

To extract the five selected class folders under the git-ignored `data/raw/` folder, add `--extract-images` to the audit command. The legacy train/test split is not used.

## Repository layout

```text
app/                  Flutter application
ml/                   PyTorch data, training, evaluation, and export code
data/manifests/       Generated dataset manifests (not committed)
reports/data_audit/   Data quality findings and review notes
docs/                 Scope, architecture, decisions, and branding
.github/workflows/    Automated Android build
```

## Next step

The current reproducible split is generated from 2,384 candidates after holding out six files in three conflicting exact-duplicate pairs. Twenty-four non-exact perceptual pairs are assigned to the same split using grouped stratification. Next, train MobileNetV3-Small using PyTorch in Google Colab. Keep the course-project scope focused: one image, one waste item, five classes, offline Android inference.
