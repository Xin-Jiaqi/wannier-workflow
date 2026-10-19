from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures/benchmark/expected-outcomes.json"
TERMINALS = ("TOP3_READY", "PARTIAL_TOPK", "NO_MODEL", "TECHNICAL_FAILED")


class BenchmarkOutcomeFixtureTests(unittest.TestCase):
    def test_expected_outcomes_fixture_schema(self) -> None:
        document = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertTrue(document["cases"])
        seen: set[str] = set()
        for item in document["cases"]:
            self.assertIn(item["expected_terminal"], TERMINALS)
            self.assertTrue(item.get("material_id"))
            self.assertTrue(item.get("rationale"))
            self.assertTrue(item.get("source_ref"))
            self.assertNotIn(item["material_id"], seen)
            seen.add(item["material_id"])

    def test_minimum_benchmark_classes_covered(self) -> None:
        document = json.loads(FIXTURE.read_text(encoding="utf-8"))
        expected = {
            "LiAuI4": "TOP3_READY",
            "PdS2": "PARTIAL_TOPK",
            "SSn": "NO_MODEL",
            "SiS2": "NO_MODEL",
            "ZrI2": "TECHNICAL_FAILED",
        }
        actual = {item["material_id"]: item["expected_terminal"] for item in document["cases"]}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
