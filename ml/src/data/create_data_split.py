"""Create a reproducible, leakage-aware 70/15/15 split.

Uses seven stratified group folds: five folds for training, one for validation,
and one for test. Non-exact perceptual-hash candidates are kept in one group
so they cannot cross splits. Byte-identical cross-label pairs must already be
held out by build_training_candidates.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def build_group_ids(
    rows: list[dict[str, str]],
    perceptual_pairs: list[dict[str, str]],
    max_distance: int = 4,
) -> dict[str, str]:
    paths = {row["relative_path"] for row in rows}
    union_find = UnionFind(sorted(paths))

    for pair in perceptual_pairs:
        distance = int(pair.get("phash_hamming_distance", 999))
        if distance > max_distance or is_true(pair.get("pixel_identical", "false")):
            continue
        left = f'{pair["class_a"].lower()}/{pair["file_a"]}'
        right = f'{pair["class_b"].lower()}/{pair["file_b"]}'
        if left in paths and right in paths:
            union_find.union(left, right)

    clusters: dict[str, list[str]] = defaultdict(list)
    for path in sorted(paths):
        clusters[union_find.find(path)].append(path)

    ordered = sorted((sorted(members) for members in clusters.values()), key=lambda x: x[0])
    return {
        path: f"G{group_index:05d}"
        for group_index, members in enumerate(ordered, start=1)
        for path in members
    }


def create_split(
    rows: list[dict[str, str]],
    perceptual_pairs: list[dict[str, str]],
    seed: int = 42,
    n_splits: int = 7,
) -> list[dict[str, str]]:
    if n_splits != 7:
        raise ValueError("FARZ uses exactly seven folds for the approximate 70/15/15 split.")
    if len(rows) < n_splits:
        raise ValueError("Not enough candidate images for a seven-fold split.")

    group_by_path = build_group_ids(rows, perceptual_pairs)
    labels = np.asarray([row["class_name"] for row in rows])
    groups = np.asarray([group_by_path[row["relative_path"]] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    assignment: dict[int, int] = {}
    for fold_number, (_, fold_indices) in enumerate(
        splitter.split(np.zeros(len(rows)), labels, groups)
    ):
        for index in fold_indices.tolist():
            if index in assignment:
                raise RuntimeError(f"Image index {index} assigned to multiple folds.")
            assignment[index] = fold_number

    if len(assignment) != len(rows):
        raise RuntimeError("The seven folds did not assign every candidate image exactly once.")

    output: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        fold_number = assignment[index]
        split_name = "test" if fold_number == 0 else "validation" if fold_number == 1 else "train"
        output.append({
            "relative_path": row["relative_path"],
            "class_name": row["class_name"],
            "sha256": row["sha256"],
            "group_id": group_by_path[row["relative_path"]],
            "fold": str(fold_number),
            "split": split_name,
        })

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in output:
        group_splits[row["group_id"]].add(row["split"])
    leaking = [group for group, split_names in group_splits.items() if len(split_names) != 1]
    if leaking:
        raise RuntimeError(f"Group leakage detected: {leaking[:5]}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()

    candidates_path = root / "data/manifests/training_candidate_manifest.csv"
    perceptual_path = root / "reports/data_audit/perceptual_duplicate_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError("Run audit_dataset.py and build_training_candidates.py first.")
    if not perceptual_path.exists():
        raise FileNotFoundError(f"Missing perceptual audit file: {perceptual_path}")

    candidates = read_csv(candidates_path)
    pairs = read_csv(perceptual_path)
    split_rows = create_split(candidates, pairs, seed=args.seed)

    output_path = root / "data/manifests/data_split.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(split_rows[0].keys()))
        writer.writeheader()
        writer.writerows(split_rows)

    counts_by_split = {}
    for split_name in ("train", "validation", "test"):
        selected = [row for row in split_rows if row["split"] == split_name]
        counts_by_split[split_name] = dict(Counter(row["class_name"] for row in selected))

    summary = {
        "seed": args.seed,
        "method": "StratifiedGroupKFold(7): folds 2-6 train, fold 1 validation, fold 0 test",
        "max_phash_distance_grouped": 4,
        "candidate_images": len(split_rows),
        "unique_groups": len({row["group_id"] for row in split_rows}),
        "split_counts": dict(Counter(row["split"] for row in split_rows)),
        "class_counts": counts_by_split,
        "group_leakage": False,
        "exact_conflict_files_included": 0,
    }
    report_path = root / "reports/data_audit/split_counts.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Split manifest: {output_path}")


if __name__ == "__main__":
    main()
