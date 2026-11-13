"""PyTorch Dataset and image transforms for FARZ's five-class TrashNet subset."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Callable
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ("Glass", "Paper", "Cardboard", "Plastic", "Metal")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class SquarePad:
    """Pad 4:3 photos to square without cropping the object."""
    def __init__(self, fill=(245, 247, 242)):
        self.fill = fill
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left, top = (side-width)//2, (side-height)//2
        right, bottom = side-width-left, side-height-top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)

def build_transforms(image_size: int = 224) -> tuple[Callable, Callable]:
    train_tf = transforms.Compose([
        SquarePad(), transforms.Resize((image_size+24, image_size+24)),
        transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        SquarePad(), transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

class FARZDataset(Dataset):
    """Load a split from data_split.csv, resolving paths under images_dir."""
    def __init__(self, manifest_path, images_dir, split: str, transform=None):
        self.manifest_path = Path(manifest_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.manifest_path}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        with self.manifest_path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.rows = [row for row in rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {self.manifest_path}")
        unknown = sorted({r["class_name"] for r in self.rows} - set(CLASS_NAMES))
        if unknown:
            raise ValueError(f"Unexpected labels in split manifest: {unknown}")
        missing = [r["relative_path"] for r in self.rows
                   if not (self.images_dir / r["relative_path"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images missing under {self.images_dir}; examples: {missing[:5]}"
            )
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        path = self.images_dir / row["relative_path"]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        except Exception as exc:
            raise RuntimeError(f"Unable to load image {path}: {exc}") from exc
        return tensor, CLASS_TO_IDX[row["class_name"]]
