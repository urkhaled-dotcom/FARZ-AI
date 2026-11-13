# FARZ — PyTorch training pipeline

The first model is **MobileNetV3-Small**, initialized from ImageNet weights and fine-tuned for five independent TrashNet material classes.

## Pipeline files
- `src/data/audit_dataset.py`: build the five-class image manifest and verify image files.
- `src/data/build_training_candidates.py`: hold out six members of three conflicting exact-duplicate pairs.
- `src/data/create_data_split.py`: reproducible stratified group split.
- `src/data/dataset.py`: PyTorch Dataset, square padding, augmentation, and normalization.
- `src/models/mobilenet.py`: MobileNetV3-Small factory.
- `src/training/train.py`: training loop, validation checkpointing, scheduler, AMP, and early stopping.
- `src/evaluation/evaluate.py`: final evaluation on the untouched test split.
- `notebooks/FARZ_Training_Colab.ipynb`: guided Google Colab notebook.

## Data folder
Extract the source archive so `data/raw/dataset-resized/{glass,paper,cardboard,plastic,metal,trash}` exists. The audit ignores Trash. Original photos are never modified or deleted.

## Rebuild the manifests
```bash
pip install -r ml/requirements.txt
python ml/src/data/audit_dataset.py --images-dir data/raw/dataset-resized --root .
python ml/src/data/build_training_candidates.py --root .
python ml/src/data/create_data_split.py --root . --seed 42
```

The current split is 1,702 train / 341 validation / 341 test. Three byte-identical pairs with conflicting labels (six files) are held out for manual review. Twenty-four additional non-identical perceptual-hash candidate pairs are kept together before splitting; they are review candidates, not confirmed duplicates.

## Train and evaluate
```bash
python ml/src/training/train.py --root . --epochs 20 --batch-size 32 --patience 5
python ml/src/evaluation/evaluate.py --root .
```

The first run uses ImageNet pretrained weights and requires internet to download them. CUDA automatically enables mixed precision. The test split is not used for checkpoint selection. Results are saved under `ml/outputs/mobilenet_v3_small/` (not committed to Git).

The score applies to the controlled-background TrashNet dataset, not arbitrary cluttered real-world scenes. FARZ's first app release handles one waste item per image.
