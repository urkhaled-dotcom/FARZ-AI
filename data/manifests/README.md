# Dataset manifests

The full dataset manifests are generated locally from the original TrashNet archive and are not committed with the source images.

- `five_class_manifest.csv`: 2,390 valid images from the five agreed classes.
- `training_candidate_manifest.csv`: 2,384 candidates after holding out six members of three exact cross-label duplicate pairs.
- `data_split.csv`: generated 70/15/15-style stratified group split (1,702 train / 341 validation / 341 test). All members of each perceptual candidate pair with pHash distance <= 4 are kept in the same split.

From the repository root, run these commands after making the original dataset ZIP available:

```bash
python ml/src/data/audit_dataset.py --archive /path/to/dataset-resized.zip --root .
python ml/src/data/build_training_candidates.py --root .
python ml/src/data/create_data_split.py --root .
```

The original archive and source images are not modified. The legacy train/test split is not used.
