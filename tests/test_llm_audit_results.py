import csv
import hashlib
import json
from pathlib import Path
import unittest


RESULTS = (
    Path(__file__).parents[1]
    / "benchmark"
    / "llm_agent_audit"
    / "fresh_locked_results_v2"
)
RUNNER = RESULTS.parent / "run_locked_audit_v2.py"


class FrozenLLMAuditResultTests(unittest.TestCase):
    def test_fresh_v2_primary_guards_are_frozen_and_pass(self):
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(
            summary["audit_version"],
            "llm_agent_memory_fresh_locked_v2",
        )
        self.assertTrue(summary["primary_pass"])
        self.assertTrue(all(summary["decisions"].values()))
        self.assertGreater(summary["paired"]["full_minus_hard"]["ci_low"], 0.0)

    def test_frozen_result_has_one_row_per_seed_and_condition(self):
        summary = json.loads((RESULTS / "summary.json").read_text())
        with (RESULTS / "per_episode.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            len(rows),
            len(summary["locked_seeds"]) * len(summary["conditions"]),
        )
        self.assertTrue(all(len(row["prompt_sha256"]) == 64 for row in rows))

    def test_frozen_runner_hash_matches_summary(self):
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(
            hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            summary["runner_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
