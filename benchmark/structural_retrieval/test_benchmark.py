import unittest

from run_benchmark import POLICIES, run_seed, summarize


class StructuralRetrievalBenchmarkTests(unittest.TestCase):
    def test_small_audit_is_matched_and_passes_mechanism_guards(self):
        rows = [
            row
            for seed in (7000, 7001, 7002)
            for row in run_seed(seed, 12)
        ]
        self.assertEqual(
            {str(row["policy"]) for row in rows},
            set(POLICIES),
        )
        summary = summarize(rows, bootstrap_samples=500)
        self.assertTrue(summary["guards"]["matched_selected_count"])
        self.assertTrue(summary["guards"]["lineage_echo_invariant"])
        self.assertGreater(
            summary["absolute"]["cycle_aware"]["accuracy"],
            summary["absolute"]["recent"]["accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
