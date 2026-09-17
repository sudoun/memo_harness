import importlib.util
from pathlib import Path
import sys
import unittest


RUNNER = (
    Path(__file__).parents[1]
    / "benchmark"
    / "llm_agent_audit"
    / "run_locked_audit.py"
)
SPEC = importlib.util.spec_from_file_location("locked_llm_audit", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LLMAuditContractTests(unittest.TestCase):
    def test_locked_seed_and_condition_contract(self):
        self.assertEqual(MODULE.SEEDS, tuple(range(3100, 3140)))
        self.assertEqual(
            MODULE.CONDITIONS,
            (
                "raw_context",
                "hard_top1_memory",
                "full_posterior_memory",
            ),
        )
        self.assertEqual(MODULE.MAX_NEW_TOKENS, 20)

    def test_parser_requires_one_unique_allowed_label(self):
        labels = MODULE.s3_law().labels
        self.assertEqual(MODULE.parse_label("The answer is 231.", labels), "231")
        self.assertIsNone(MODULE.parse_label("231 or 312", labels))
        self.assertIsNone(MODULE.parse_label("unknown", labels))

    def test_prompt_has_no_ground_truth_field(self):
        episode = MODULE.generate_episode(MODULE.SEEDS[0])
        messages = MODULE.make_messages(episode, "raw_context", None)
        text = "\n".join(message["content"] for message in messages)
        self.assertNotIn("true_label", text)
        self.assertNotIn("ground truth", text.lower())

    def test_capsules_come_from_same_observations_but_different_compression(self):
        episode = MODULE.generate_episode(MODULE.SEEDS[0])
        hard, full = MODULE.build_capsules(episode)
        self.assertEqual(
            hard["used_observation_ids"],
            full["used_observation_ids"],
        )
        self.assertEqual(hard["inference_backend"], "exact")
        self.assertEqual(full["inference_backend"], "exact")


if __name__ == "__main__":
    unittest.main()
