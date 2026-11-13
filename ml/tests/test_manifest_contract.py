"""Data contract tests for the agreed FARZ five-class subset."""
import csv
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifests" / "five_class_manifest.csv"
CANDIDATES = ROOT / "data" / "manifests" / "training_candidate_manifest.csv"
EXPECTED = {"Glass": 501, "Paper": 594, "Cardboard": 403, "Plastic": 482, "Metal": 410}


class ManifestContractTest(unittest.TestCase):
    def test_manifest_matches_agreed_five_classes(self):
        self.assertTrue(MANIFEST.exists(), "Run audit_dataset.py first")
        with MANIFEST.open(newline="", encoding="utf-8") as file:
            counts = Counter(row["class_name"] for row in csv.DictReader(file))
        self.assertEqual(dict(counts), EXPECTED)
        self.assertEqual(sum(counts.values()), 2390)

    def test_conflicting_exact_duplicates_are_held_out(self):
        self.assertTrue(CANDIDATES.exists(), "Run build_training_candidates.py first")
        with CANDIDATES.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2384)
        self.assertEqual(len({row["sha256"] for row in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
