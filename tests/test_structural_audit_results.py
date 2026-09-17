import csv
import hashlib
import json
from pathlib import Path
import unittest


RESULTS = (
    Path(__file__).parents[1]
    / "benchmark"
    / "structural_retrieval"
    / "formal_results"
)
RUNNER = RESULTS.parent / "run_benchmark.py"


class FrozenStructuralRetrievalResultTests(unittest.TestCase):
    def test_formal_guards_pass_under_matched_budget(self):
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertTrue(summary["passed"])
        self.assertTrue(all(summary["guards"].values()))
        selected = {
            values["selected"]
            for values in summary["absolute"].values()
        }
        self.assertEqual(selected, {summary["observation_budget"]})
        self.assertGreater(
            summary["paired_cycle_aware_minus_baseline"]["recent"][
                "accuracy"
            ]["ci95"][0],
            0.0,
        )
        self.assertLess(
            summary["paired_cycle_aware_minus_baseline"]["recent"][
                "nll"
            ]["ci95"][1],
            0.0,
        )

    def test_formal_result_has_twenty_paired_seed_rows_per_policy(self):
        with (RESULTS / "per_seed.csv").open(
            encoding="utf-8",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        self.assertEqual(
            {row["policy"] for row in rows},
            {"recent", "coverage", "cycle_aware"},
        )
        self.assertEqual(len({row["seed"] for row in rows}), 20)

    def test_runner_and_result_hashes_match_provenance(self):
        provenance = json.loads(
            (RESULTS / "PROVENANCE.json").read_text()
        )
        self.assertEqual(
            hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            provenance["runner_sha256"],
        )
        for name, expected in provenance["result_sha256"].items():
            self.assertEqual(
                hashlib.sha256((RESULTS / name).read_bytes()).hexdigest(),
                expected,
            )
        root = Path(__file__).parents[1]
        for name, expected in provenance[
            "package_source_sha256"
        ].items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
