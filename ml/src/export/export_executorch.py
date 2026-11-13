"""Export FARZ MobileNetV3-Small to ExecuTorch/XNNPACK and verify runtime parity."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    parser.add_argument("--parity-images", type=int, default=32)
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
    if args.parity_images < 1:
        raise ValueError("--parity-images must be at least 1")

    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.runtime import Runtime
    except Exception as exc:
        raise RuntimeError(
            "ExecuTorch is not installed or its XNNPACK export/runtime modules are unavailable. "
            "Install the pinned exporter version with: pip install executorch==1.5.1"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_classes = tuple(checkpoint.get("class_names", ()))
    if checkpoint_classes != tuple(CLASS_NAMES):
        raise ValueError(
            f"Checkpoint class order mismatch. Expected {CLASS_NAMES}; got {checkpoint_classes}"
        )

    model = build_model(pretrained=False).cpu().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    image_size = int(checkpoint.get("image_size", 224))
    _, eval_transform = build_transforms(image_size)
    test_dataset = FARZDataset(manifest_path, images_dir, "test", eval_transform)
    sample_tensor, _ = test_dataset[0]
    example_input = sample_tensor.unsqueeze(0).contiguous()

    print(f"PyTorch: {torch.__version__}")
    try:
        import executorch
        print(f"ExecuTorch: {getattr(executorch, '__version__', 'installed')}")
    except Exception:
        print("ExecuTorch: installed")
    print(f"Input: {tuple(example_input.shape)} | classes: {CLASS_NAMES}")

    with torch.inference_mode():
        pytorch_sample = model(example_input)

    # Stage 1: validate the PyTorch export graph before backend lowering.
    exported_program = torch.export.export(model, (example_input,), strict=True)
    with torch.inference_mode():
        exported_output = exported_program.module()(example_input)
    torch.testing.assert_close(exported_output, pytorch_sample, rtol=0, atol=0)
    print(f"torch.export graph validated ({len(list(exported_program.graph.nodes))} nodes).")

    # Stage 2: lower to the XNNPACK CPU backend used by the Android AAR.
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        file.write(executorch_program.buffer)
    print(f"Saved PTE: {output_path} ({output_path.stat().st_size:,} bytes)")

    # Stage 3: load the PTE through the ExecuTorch Python runtime and compare
    # class predictions against the original PyTorch model on held-out images.
    runtime_program = Runtime.get().load_program(str(output_path))
    method = runtime_program.load_method("forward")
    count = min(args.parity_images, len(test_dataset))
    max_abs_error = 0.0
    max_rel_error = 0.0
    matched = 0
    with torch.inference_mode():
        for index in range(count):
            image_tensor, _ = test_dataset[index]
            input_batch = image_tensor.unsqueeze(0).contiguous()
            pytorch_logits = model(input_batch)
            executorch_logits = method.execute((input_batch,))[0]
            if not isinstance(executorch_logits, torch.Tensor):
                executorch_logits = torch.as_tensor(executorch_logits)
            difference = (pytorch_logits - executorch_logits).abs()
            max_abs_error = max(max_abs_error, float(difference.max().item()))
            denominator = pytorch_logits.abs().clamp_min(1e-8)
            max_rel_error = max(max_rel_error, float((difference / denominator).max().item()))
            if int(pytorch_logits.argmax(1).item()) == int(executorch_logits.argmax(1).item()):
                matched += 1

    prediction_match_rate = matched / count
    report = {
        "model": "MobileNetV3-Small",
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "class_names": list(CLASS_NAMES),
        "input": {
            "layout": "NCHW",
            "shape": [1, 3, image_size, image_size],
            "color": "RGB",
            "resize": "square-pad then bilinear resize",
            "padding_rgb": [245, 247, 242],
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "backend": "XNNPACK",
        "executorch_pte": output_path.name,
        "pte_size_bytes": output_path.stat().st_size,
        "parity_test_images": count,
        "top1_prediction_matches": matched,
        "top1_prediction_match_rate": prediction_match_rate,
        "max_absolute_logit_error": max_abs_error,
        "max_relative_logit_error": max_rel_error,
        "test_split_used_for_export_selection": False,
        "note": "Softmax scores are not calibrated probabilities.",
    }

    report_path = output_path.with_name("executorch_export_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if prediction_match_rate < 1.0:
        raise RuntimeError(
            f"ExecuTorch top-1 parity failed: {matched}/{count} predictions match. "
            "Keep the checkpoint; inspect export/runtime version and operator lowering."
        )

    metadata_path = output_path.with_name("farz_model_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.copy_to_app:
        app_models = root / "app/assets/models"
        app_models.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, app_models / "farz_model.pte")
        shutil.copy2(metadata_path, app_models / "farz_model_metadata.json")
        print(f"Copied verified model assets to: {app_models}")

    print("Export and ExecuTorch Python runtime parity check completed.")


if __name__ == "__main__":
    main()
