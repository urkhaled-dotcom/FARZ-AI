"""Audit the five-class TrashNet subset without modifying source archives.

Accepts the original trashnet-master.zip (containing the nested
data/dataset-resized.zip) or dataset-resized.zip directly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

CLASSES = ("glass", "paper", "cardboard", "plastic", "metal")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dataset_archive(archive_path: Path) -> zipfile.ZipFile:
    outer = zipfile.ZipFile(archive_path)
    nested_name = next(
        (name for name in outer.namelist() if name.endswith("/data/dataset-resized.zip")),
        None,
    )
    if nested_name is None:
        if any(name.startswith("dataset-resized/") for name in outer.namelist()):
            return outer
        outer.close()
        raise ValueError("Could not locate data/dataset-resized.zip in the supplied archive.")
    nested_bytes = outer.read(nested_name)
    outer.close()
    return zipfile.ZipFile(io.BytesIO(nested_bytes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--root", default=Path.cwd(), type=Path)
    parser.add_argument("--extract-images", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = root / "data/manifests/five_class_manifest.csv"
    duplicates_path = root / "reports/data_audit/exact_duplicates.csv"
    counts_path = root / "reports/data_audit/class_counts.json"
    extract_root = root / "data/raw/trashnet_five_class"
    for path in (manifest_path.parent, duplicates_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    if args.extract_images:
        extract_root.mkdir(parents=True, exist_ok=True)

    zip_file = find_dataset_archive(args.archive)
    rows: list[dict[str, object]] = []
    hash_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    invalid: list[str] = []

    with zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if path.name.startswith("._") or path.name == ".DS_Store":
                continue
            parts = path.parts
            if len(parts) < 3 or parts[0] != "dataset-resized":
                continue
            class_name = parts[1].lower()
            if class_name not in CLASSES or path.suffix.lower() not in IMAGE_EXTS:
                continue

            payload = zip_file.read(member)
            digest = hashlib.sha256(payload).hexdigest()
            width = height = None
            mode = "unknown"
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                with Image.open(io.BytesIO(payload)) as image:
                    width, height, mode = image.width, image.height, image.mode
            except Exception:
                invalid.append(member.filename)

            row = {
                "archive_path": member.filename,
                "relative_path": "/".join(parts[1:]),
                "class_name": class_name.title(),
                "file_name": path.name,
                "sha256": digest,
                "bytes": len(payload),
                "width": width,
                "height": height,
                "mode": mode,
                "valid_image": member.filename not in invalid,
            }
            rows.append(row)
            hash_groups[digest].append(row)

            if args.extract_images:
                output = extract_root / class_name / path.name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)

    rows.sort(key=lambda row: (str(row["class_name"]), str(row["file_name"])))
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    duplicate_rows = []
    for digest, group in hash_groups.items():
        if len(group) < 2:
            continue
        labels = sorted({str(row["class_name"]) for row in group})
        kind = "cross_label_conflict" if len(labels) > 1 else "same_class_duplicate"
        for row in group:
            duplicate_rows.append({
                "sha256": digest,
                "duplicate_type": kind,
                "class_name": row["class_name"],
                "relative_path": row["relative_path"],
                "all_labels_in_group": "|".join(labels),
            })
    with duplicates_path.open("w", newline="", encoding="utf-8") as file:
        columns = ["sha256", "duplicate_type", "class_name", "relative_path", "all_labels_in_group"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(duplicate_rows)

    class_counts = Counter(str(row["class_name"]) for row in rows)
    report = {
        "class_counts": dict(class_counts),
        "total_five_class_images": len(rows),
        "invalid_images": invalid,
        "exact_duplicate_groups": sum(len(group) > 1 for group in hash_groups.values()),
        "cross_label_duplicate_groups": sum(
            len(group) > 1 and len({str(row["class_name"]) for row in group}) > 1
            for group in hash_groups.values()
        ),
        "source_was_modified": False,
    }
    counts_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Exact duplicates: {duplicates_path}")
    if invalid:
        raise SystemExit("Invalid images found; review the audit report before training.")


if __name__ == "__main__":
    main()
