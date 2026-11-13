"""Export the trained FARZ checkpoint to a PyTorch Mobile Lite model.

Use the matching host stack: torch==2.1.0+cpu and torchvision==0.16.0+cpu.
The output is a .ptl Lite Interpreter archive for pytorch_android_lite:2.1.0.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.data.dataset import (
    CLASS_NAMES,
    FARZDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
)
from ml.src.models.mobilenet import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--parity-images", type=int, default=30)
    parser.add_argument("--copy-to-app", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    checkpoint_path = (args.checkpoint or root / "ml/outputs/mobilenet_v3_small/best_model.pth").resolve()
    manifest_path = (args.manifest or root / "data/manifests/data_split_run_01.csv").resolve()
    images_dir = (args.images_dir or root / "data/raw/dataset-resized").resolve()
    output_path = (args.output or root / "ml/outputs/mobilenet_v3_small/farz_model.ptl").resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if args.parity_images < len(CLASS_NAMES):
        raise ValueError("--parity-images must be at least 5 (one image per class).")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if tuple(checkpoint.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError(f"Unexpected model class order: {checkpoint.get('class_names')}")

    model = build_model(pretrained=False).cpu().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    image_size = int(checkpoint.get("image_size", 224))
    _, eval_transform = build_transforms(image_size)
    test_dataset = FARZDataset(manifest_path, images_dir, "test", eval_transform)

    sample, _ = test_dataset[0]
    example_input = (sample.unsqueeze(0).contiguous(),)
    with torch.inference_mode():
        eager_logits = model(*example_input)

    traced = torch.jit.trace(model, example_input, strict=True)
    frozen = torch.jit.freeze(traced.eval())
    lite_module = optimize_for_mobile(frozen, backend="CPU")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lite_module._save_for_lite_interpreter(str(output_path))
    print(f"Wrote Lite model: {output_path} ({output_path.stat().st_size:,} bytes)")

    # Check top-1 parity on a balanced sample across all five classes.
    by_class = {name: [] for name in CLASS_NAMES}
    for index, row in enumerate(test_dataset.rows):
        by_class[row["class_name"]].append(index)
    per_class = max(1, args.parity_images // len(CLASS_NAMES))
    selected = []
    for name in CLASS_NAMES:
        selected.extend(by_class[name][:per_class])
    if len(selected) < args.parity_images:
        selected_set = set(selected)
        selected.extend(i for i in range(len(test_dataset)) if i not in selected_set)
        selected = selected[:args.parity_images]

    matches = 0
    max_absolute_error = 0.0
    parity_rows = []
    with torch.inference_mode():
        for index in selected:
            image_tensor, _ = test_dataset[index]
            batch = image_tensor.unsqueeze(0).contiguous()
            expected = model(batch)
            actual = lite_module(batch)
            difference = (expected - actual).abs()
            max_absolute_error = max(max_absolute_error, float(difference.max().item()))
            expected_index = int(expected.argmax(1).item())
            actual_index = int(actual.argmax(1).item())
            matched = expected_index == actual_index
            matches += int(matched)
            parity_rows.append({
                "relative_path": test_dataset.rows[index]["relative_path"],
                "pytorch_top1": CLASS_NAMES[expected_index],
                "lite_top1": CLASS_NAMES[actual_index],
                "match": matched,
            })

    match_rate = matches / len(selected)
    report = {
        "model": "MobileNetV3-Small",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "input_layout": "NCHW",
        "color_order": "RGB",
        "padding_rgb": [245, 247, 242],
        "resize": "224x224 bilinear after centered square padding",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "runtime": "org.pytorch:pytorch_android_lite:2.1.0",
        "host_torch": torch.__version__,
        "parity_images": len(selected),
        "top1_matches": matches,
        "top1_match_rate": match_rate,
        "max_absolute_logit_error": max_absolute_error,
        "note": "Python Lite module parity is necessary but does not replace physical Android device testing.",
    }
    output_path.with_name("pytorch_lite_export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    output_path.with_name("pytorch_lite_parity_predictions.json").write_text(
        json.dumps(parity_rows, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_name": "FARZ MobileNetV3-Small",
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "input_layout": "NCHW",
        "color_order": "RGB",
        "padding_rgb": [245, 247, 242],
        "resize_interpolation": "bilinear",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "runtime": "PyTorch Android Lite 2.1.0",
        "backend": "CPU",
    }
    metadata_path = output_path.with_name("farz_model_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if match_rate != 1.0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Lite model parity failed: {matches}/{len(selected)} predictions matched. "
            "Do not bundle this model in the Android app."
        )

    if args.copy_to_app:
        app_assets = root / "app/assets/models"
        app_assets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, app_assets / "farz_model.ptl")
        shutil.copy2(metadata_path, app_assets / "farz_model_metadata.json")
        print("Copied verified Lite model and metadata to", app_assets)

    print("Lite export and Python parity passed. Physical Android validation is still required.")


if __name__ == "__main__":
    main()
