"""Build a conservative candidate manifest without deleting source files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    manifest = read_csv(root / "data/manifests/five_class_manifest.csv")
    duplicates = read_csv(root / "reports/data_audit/exact_duplicates.csv")
    conflict_paths = {
        row["relative_path"]
        for row in duplicates
        if row["duplicate_type"] == "cross_label_conflict"
    }

    held_out = [row for row in manifest if row["relative_path"] in conflict_paths]
    candidates = [row for row in manifest if row["relative_path"] not in conflict_paths]
    if candidates:
        destination = root / "data/manifests/training_candidate_manifest.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(candidates[0].keys()))
            writer.writeheader()
            writer.writerows(candidates)

    review_path = root / "reports/data_audit/held_out_conflicting_duplicates.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["class_name", "relative_path", "sha256", "reason"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in held_out:
            writer.writerow({
                "class_name": row["class_name"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "reason": "byte-identical content appears under conflicting class labels; held out for manual review",
            })

    summary = {
        "full_manifest_images": len(manifest),
        "training_candidate_images": len(candidates),
        "held_out_images": len(held_out),
        "training_candidate_class_counts": dict(Counter(row["class_name"] for row in candidates)),
        "physical_source_images_deleted_or_modified": 0,
        "split_created": False,
    }
    report_path = root / "reports/data_audit/training_candidate_counts.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
