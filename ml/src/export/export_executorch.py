"""Export FARZ MobileNetV3-Small to ExecuTorch/XNNPACK and verify parity."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.data.dataset import (
    CLASS_NAMES, FARZDataset, IMAGENET_MEAN, IMAGENET_STD, build_transforms,
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
    manifest_path = (args.manifest or root / "data/manifests/data_split.csv").resolve()
    images_dir = (args.images_dir or root / "data/raw/dataset-resized").resolve()
    output_path = (args.output or root / "ml/outputs/mobilenet_v3_small/farz_model.pte").resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if args.parity_images < len(CLASS_NAMES):
        raise ValueError("--parity-images must be at least 5 (one image per class).")

    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.runtime import Runtime
    except Exception as exc:
        raise RuntimeError(
            "ExecuTorch exporter/runtime is unavailable. Install the Android-matched version: "
            "pip install executorch==1.5.1"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        expected = model(*example_input)
    exported_program = torch.export.export(model, example_input, strict=True)
    with torch.inference_mode():
        actual = exported_program.module()(*example_input)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("torch.export graph validated:", len(list(exported_program.graph.nodes)), "nodes")

    edge_program = to_edge_transform_and_lower(
        exported_program, partitioner=[XnnpackPartitioner()]
    )
    executorch_program = edge_program.to_executorch()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(executorch_program.buffer)
    print(f"Wrote PTE: {output_path} ({output_path.stat().st_size:,} bytes)")

    runtime_program = Runtime.get().load_program(str(output_path))
    method = runtime_program.load_method("forward")

    # Use a balanced sample so every class participates in the export parity gate.
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

    max_absolute_error = 0.0
    max_relative_error = 0.0
    matches = 0
    parity_rows = []
    with torch.inference_mode():
        for index in selected:
            image_tensor, _ = test_dataset[index]
            batch = image_tensor.unsqueeze(0).contiguous()
            pytorch_logits = model(batch)
            executorch_logits = method.execute((batch,))[0]
            if not isinstance(executorch_logits, torch.Tensor):
                executorch_logits = torch.as_tensor(executorch_logits)
            difference = (pytorch_logits - executorch_logits).abs()
            max_absolute_error = max(max_absolute_error, float(difference.max().item()))
            relative_error = difference / pytorch_logits.abs().clamp_min(1e-8)
            max_relative_error = max(max_relative_error, float(relative_error.max().item()))
            pytorch_index = int(pytorch_logits.argmax(1).item())
            executorch_index = int(executorch_logits.argmax(1).item())
            is_match = pytorch_index == executorch_index
            matches += int(is_match)
            parity_rows.append({
                "relative_path": test_dataset.rows[index]["relative_path"],
                "pytorch_top1": CLASS_NAMES[pytorch_index],
                "executorch_top1": CLASS_NAMES[executorch_index],
                "match": is_match,
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
        "backend": "XNNPACK",
        "executorch_version": "1.5.1",
        "android_runtime": "org.pytorch:executorch-android:1.5.1",
        "pte_file": output_path.name,
        "pte_size_bytes": output_path.stat().st_size,
        "parity_images": len(selected),
        "top1_matches": matches,
        "top1_match_rate": match_rate,
        "max_absolute_logit_error": max_absolute_error,
        "max_relative_logit_error": max_relative_error,
        "test_split_used_for_model_selection": False,
        "note": "Python ExecuTorch parity is not a substitute for testing the PTE on a physical Android device.",
    }
    output_path.with_name("executorch_export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    output_path.with_name("executorch_parity_predictions.json").write_text(
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
        "runtime": "ExecuTorch Android 1.5.1",
        "backend": "XNNPACK",
    }
    metadata_path = output_path.with_name("farz_model_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if match_rate != 1.0:
        raise RuntimeError(
            f"PTE parity failed: {matches}/{len(selected)} top-1 predictions matched. "
            "Do not bundle this PTE in the Android app."
        )

    if args.copy_to_app:
        app_assets = root / "app/assets/models"
        app_assets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, app_assets / "farz_model.pte")
        shutil.copy2(metadata_path, app_assets / "farz_model_metadata.json")
        print("Copied verified PTE and metadata to", app_assets)

    print("Export and Python runtime parity passed. Physical Android validation is still required.")


if __name__ == "__main__":
    main()
