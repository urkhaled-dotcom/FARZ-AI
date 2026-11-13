"""Dataset contract checks. Manifest-dependent checks skip until the Colab audit runs."""
import csv
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifests" / "five_class_manifest.csv"
CANDIDATES = ROOT / "data" / "manifests" / "training_candidate_manifest.csv"
EXPECTED = {"Glass": 501, "Paper": 594, "Cardboard": 403, "Plastic": 482, "Metal": 410}


class ManifestContractTest(unittest.TestCase):
    def test_five_class_contract(self):
        self.assertEqual(list(EXPECTED), ["Glass", "Paper", "Cardboard", "Plastic", "Metal"])
        self.assertEqual(sum(EXPECTED.values()), 2390)

    @unittest.skipUnless(MANIFEST.exists(), "Run the Colab data audit to create manifests")
    def test_generated_manifest_matches_agreed_counts(self):
        with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
            counts = Counter(row["class_name"] for row in csv.DictReader(file))
        self.assertEqual(dict(counts), EXPECTED)
        self.assertEqual(sum(counts.values()), 2390)

    @unittest.skipUnless(CANDIDATES.exists(), "Run build_training_candidates.py to create candidates")
    def test_conflicting_exact_duplicates_are_held_out(self):
        with CANDIDATES.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2384)
        self.assertEqual(len({row["sha256"] for row in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
