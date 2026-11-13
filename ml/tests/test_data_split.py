"""Unit tests for leakage-aware grouped splitting."""
from collections import defaultdict

from ml.src.data.create_data_split import build_group_ids, create_split


def synthetic_data():
    classes = ["Glass", "Paper", "Cardboard", "Plastic", "Metal"]
    rows = []
    pairs = []
    for class_name in classes:
        class_key = class_name.lower()
        for index in range(14):
            rows.append({
                "relative_path": f"{class_key}/{class_key}_{index}.jpg",
                "class_name": class_name,
                "sha256": f"{class_key}_{index}",
            })
        pairs.append({
            "phash_hamming_distance": "2",
            "pixel_identical": "False",
            "class_a": class_key,
            "file_a": f"{class_key}_0.jpg",
            "class_b": class_key,
            "file_b": f"{class_key}_1.jpg",
        })
    return rows, pairs


def test_near_duplicate_pairs_stay_in_the_same_split():
    rows, pairs = synthetic_data()
    result = create_split(rows, pairs, seed=42)
    group_splits = defaultdict(set)
    for row in result:
        group_splits[row["group_id"]].add(row["split"])

    assert len(result) == 70
    assert all(len(split_names) == 1 for split_names in group_splits.values())
    assert {row["split"] for row in result} == {"train", "validation", "test"}
    assert len(group_splits) == 65


def test_exact_duplicate_pairs_are_not_added_to_near_duplicate_groups():
    rows, _ = synthetic_data()
    pair = [{
        "phash_hamming_distance": "0",
        "pixel_identical": "True",
        "class_a": "glass",
        "file_a": "glass_0.jpg",
        "class_b": "glass",
        "file_b": "glass_1.jpg",
    }]
    groups = build_group_ids(rows, pair)
    assert groups["glass/glass_0.jpg"] != groups["glass/glass_1.jpg"]
