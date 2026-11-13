# FARZ — PyTorch training pipeline

The first model is **MobileNetV3-Small**, initialized from ImageNet weights and fine-tuned for five independent TrashNet material classes.

## Files
- `src/data/audit_dataset.py`: build the five-class image manifest and verify files.
- `src/data/build_training_candidates.py`: hold out six members of three conflicting exact-duplicate pairs.
- `src/data/create_data_split.py`: reproducible stratified group split.
- `src/data/dataset.py`: PyTorch Dataset, square padding, augmentation, and normalization.
- `src/models/mobilenet.py`: MobileNetV3-Small factory.
- `src/training/train.py`: training, validation checkpointing, scheduler, AMP, and early stopping.
- `src/evaluation/evaluate.py`: final evaluation on the untouched test split.
- `notebooks/FARZ_Training_Colab.ipynb`: guided Colab notebook.

## Download and prepare data
The notebook downloads the 42.8 MB resized TrashNet archive directly into the Colab runtime; it is not necessary to transfer it through your laptop. Extract it to `data/raw/dataset-resized`. The audit ignores Trash and never modifies or deletes source photos.

## Rebuild manifests
```bash
pip install -r ml/requirements.txt
python ml/src/data/audit_dataset.py --images-dir data/raw/dataset-resized --root .
python ml/src/data/build_training_candidates.py --root .
python ml/src/data/create_data_split.py --root . --seed 42
```

The reproducible split currently contains 1,702 train / 341 validation / 341 test images. Three byte-identical pairs with conflicting labels (six photos) are held out pending manual review. Twenty-four additional non-identical perceptual-hash candidates are grouped before splitting; those are review candidates, not confirmed duplicates.

## Train and evaluate
```bash
python ml/src/training/train.py --root . --epochs 20 --batch-size 32 --patience 5
python ml/src/evaluation/evaluate.py --root .
```

The first training run uses ImageNet pretrained weights and needs internet access to download them. CUDA automatically enables mixed precision. The test split is not used for checkpoint selection. Outputs are written under `ml/outputs/mobilenet_v3_small/` and excluded from Git.

The metrics measure performance on TrashNet's controlled-background photos, not arbitrary cluttered real-world scenes. FARZ's first app release handles one waste item per image.
