# Dataset manifests

The full dataset manifests are generated locally from the original TrashNet archive and are not committed with the source images.

- `five_class_manifest.csv`: expected to contain all 2,390 valid images from the five agreed classes.
- `training_candidate_manifest.csv`: conservative candidate pool with six members of three cross-label exact duplicate pairs held out pending manual label review.

Run `ml/src/data/audit_dataset.py` followed by `ml/src/data/build_training_candidates.py` to regenerate the manifests.

The candidate manifest is not a train/validation/test split. Do not train until remaining near-duplicate candidates and label quality are reviewed and a leakage-aware split is created.
