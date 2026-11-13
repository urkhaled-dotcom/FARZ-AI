"""Export trained FARZ MobileNetV3-Small checkpoint to ONNX and verify ORT parity."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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
    manifest_path = (args.manifest or root / "data/manifests/data_split_run_01.csv").resolve()
    images_dir = (args.images_dir or root / "data/raw/dataset-resized").resolve()
    output_path = (args.output or root / "ml/outputs/mobilenet_v3_small/farz_model.onnx").resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if args.parity_images < len(CLASS_NAMES):
        raise ValueError("--parity-images must be at least 5 (one image per class).")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError(f"Unexpected model class order: {checkpoint.get('class_names')}")

    model = build_model(pretrained=False).cpu().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    image_size = int(checkpoint.get("image_size", 224))
    _, eval_transform = build_transforms(image_size)
    test_dataset = FARZDataset(manifest_path, images_dir, "test", eval_transform)

    sample, _ = test_dataset[0]
    example = sample.unsqueeze(0).contiguous()
    with torch.inference_mode():
        eager_logits = model(example).cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        verbose=False,
        dynamo=False,
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    print(f"Wrote ONNX model: {output_path} ({output_path.stat().st_size:,} bytes)")

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if input_name != "input":
        raise RuntimeError(f"Unexpected ONNX input name: {input_name}")

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
            pytorch_logits = model(batch).cpu().numpy()
            onnx_logits = session.run([output_name], {input_name: batch.cpu().numpy()})[0]
            difference = np.abs(pytorch_logits - onnx_logits)
            max_absolute_error = max(max_absolute_error, float(difference.max()))
            pytorch_index = int(pytorch_logits.argmax(axis=1)[0])
            onnx_index = int(onnx_logits.argmax(axis=1)[0])
            matched = pytorch_index == onnx_index
            matches += int(matched)
            parity_rows.append({
                "relative_path": test_dataset.rows[index]["relative_path"],
                "pytorch_top1": CLASS_NAMES[pytorch_index],
                "onnxruntime_top1": CLASS_NAMES[onnx_index],
                "match": matched,
            })

    match_rate = matches / len(selected)
    report = {
        "model": "MobileNetV3-Small",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "input_name": input_name,
        "output_name": output_name,
        "input_layout": "NCHW",
        "color_order": "RGB",
        "padding_rgb": [245, 247, 242],
        "resize": "224x224 bilinear after centered square padding",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "onnx_opset": 17,
        "onnxruntime_version": ort.__version__,
        "android_runtime": "com.microsoft.onnxruntime:onnxruntime-android:1.31.1",
        "onnx_file": output_path.name,
        "onnx_size_bytes": output_path.stat().st_size,
        "parity_images": len(selected),
        "top1_matches": matches,
        "top1_match_rate": match_rate,
        "max_absolute_logit_error": max_absolute_error,
        "test_split_used_for_model_selection": False,
        "note": "Python ONNX Runtime parity does not replace physical Android device validation.",
    }
    output_path.with_name("onnx_export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    output_path.with_name("onnx_parity_predictions.json").write_text(
        json.dumps(parity_rows, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_name": "FARZ MobileNetV3-Small",
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "input_name": input_name,
        "output_name": output_name,
        "input_layout": "NCHW",
        "color_order": "RGB",
        "padding_rgb": [245, 247, 242],
        "resize_interpolation": "bilinear",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "runtime": "ONNX Runtime Android 1.31.1",
        "execution_provider": "CPU",
    }
    metadata_path = output_path.with_name("farz_model_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if match_rate != 1.0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ONNX parity failed: {matches}/{len(selected)} top-1 predictions matched. "
            "Do not bundle this model in the Android app."
        )

    if args.copy_to_app:
        app_assets = root / "app/assets/models"
        app_assets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, app_assets / "farz_model.onnx")
        shutil.copy2(metadata_path, app_assets / "farz_model_metadata.json")
        print("Copied verified ONNX model and metadata to", app_assets)

    print("ONNX export and Python Runtime parity passed. Physical Android validation is still required.")


if __name__ == "__main__":
    main()
