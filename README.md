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
- [x] Flutter home-screen and theme starter created.
- [x] Exact cross-label duplicate pairs documented and held out pending manual review.
- [ ] Review remaining near-duplicate candidates and label quality.
- [ ] Create leakage-aware train/validation/test split.
- [ ] Train and evaluate a compact PyTorch model.
- [ ] Validate ExecuTorch export and Android runtime.
- [ ] Integrate camera/gallery and offline inference.
- [ ] Build and test Android APK on a physical device.

## Data integrity note

The original archive must remain unchanged. Three byte-identical image pairs were found under conflicting labels; all six files are excluded from the provisional training-candidate manifest until manual adjudication. See `reports/data_audit/LABEL_REVIEW_NOTES.md`. No file is deleted or relabelled automatically.

The source dataset is not committed to this repository. Keep it in private storage or Google Drive and regenerate manifests using the audit script.

## Repository layout

```text
app/                  Flutter application
ml/                   PyTorch data, training, evaluation, and export code
reports/data_audit/   Data quality findings and review notes
docs/                 Scope, architecture, decisions, and branding
.github/workflows/    Automated Android build
```

## Next step

Complete the label/duplicate review, create a reproducible leakage-aware data split, then train the baseline model in PyTorch. Keep the course-project scope focused: one image, one waste item, five classes, offline Android inference.
