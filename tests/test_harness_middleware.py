import unittest

from posterior_memory_harness import (
    MemoryMiddleware,
    PosteriorMemoryHarness,
    cyclic_law,
)
from posterior_memory_harness.middleware import HarnessEvent, StructuredObservationEncoder


class HarnessIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.harness = PosteriorMemoryHarness().register_relation(
            "task-phase", cyclic_law(3)
        )

    @staticmethod
    def payload(identifier="o1"):
        return {
            "observation_id": identifier,
            "namespace": "run-42",
            "relation_type": "task-phase",
            "source": "checkpoint-0",
            "target": "checkpoint-1",
            "posterior": {"0": 0.02, "1": 0.96, "2": 0.02},
            "prior": {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
            "evidence_family": "tool-call-7",
            "provenance": {"tool": "tracker"},
            "observed_at": 10.0,
            "tags": ["project-x"],
        }

    @staticmethod
    def query():
        return {
            "namespace": "run-42",
            "relation_type": "task-phase",
            "nodes": ["checkpoint-0", "checkpoint-1"],
            "root": "checkpoint-0",
            "as_of": 11.0,
            "tags": ["project-x"],
        }

    def test_capsule_is_structured_and_not_an_instruction_channel(self):
        self.harness.observe_payload(self.payload())
        capsule = self.harness.before_model(self.query())
        self.assertEqual(capsule["instructions"], [])
        self.assertEqual(capsule["beliefs"][1]["candidates"][0]["state"], "1")
        self.assertEqual(capsule["used_observation_ids"], ["o1"])
        self.assertEqual(capsule["inference_diagnostics"]["mode"], "exact")
        self.assertTrue(capsule["inference_diagnostics"]["stable"])

    def test_empty_memory_returns_prior_only_capsule(self):
        capsule = self.harness.before_model(
            {
                **self.query(),
                "node_priors": {
                    "checkpoint-1": {"0": 0.1, "1": 0.2, "2": 0.7}
                },
            }
        )
        self.assertEqual(capsule["inference_backend"], "prior-only")
        self.assertEqual(capsule["used_observation_ids"], [])
        self.assertEqual(capsule["instructions"], [])
        self.assertEqual(capsule["beliefs"][0]["candidates"][0]["state"], "0")
        self.assertEqual(capsule["beliefs"][1]["candidates"][0]["state"], "2")
        self.assertEqual(
            capsule["inference_diagnostics"]["mode"], "prior-only"
        )

    def test_structured_middleware_wraps_any_request_mapping(self):
        middleware = MemoryMiddleware(
            self.harness, encoder=StructuredObservationEncoder()
        )
        identifiers = middleware.after_event(
            HarnessEvent("tool_result", {"memory_observations": [self.payload()]})
        )
        self.assertEqual(identifiers, ["o1"])
        request = middleware.before_model({"messages": []}, self.query())
        self.assertIn("memory_capsule", request)
        self.assertEqual(request["messages"], [])

    def test_revoke_is_as_of_aware(self):
        self.harness.observe_payload(self.payload())
        self.harness.revoke("o1", 12.0)
        self.assertEqual(
            self.harness.before_model(self.query())["used_observation_ids"], ["o1"]
        )
        future = {**self.query(), "as_of": 13.0}
        capsule = self.harness.before_model(future)
        self.assertEqual(capsule["used_observation_ids"], [])
        self.assertEqual(capsule["inference_backend"], "prior-only")

    def test_compaction_cannot_be_promoted_to_independent_evidence(self):
        middleware = MemoryMiddleware(
            self.harness, encoder=StructuredObservationEncoder()
        )
        compacted = {
            **self.payload("summary"),
            "evidence_family": "summary-of-tool-call-7",
            "independent_evidence": True,
        }
        middleware.on_compaction({"memory_observations": [compacted]})
        capsule = self.harness.before_model(self.query())
        self.assertEqual(capsule["used_observation_ids"], [])
        self.assertEqual(capsule["inference_backend"], "prior-only")

    def test_seed_query_discovers_a_bounded_memory_subgraph(self):
        first = self.payload("first")
        first["evidence_family"] = "first"
        second = {
            **self.payload("second"),
            "source": "checkpoint-1",
            "target": "checkpoint-2",
            "evidence_family": "second",
            "observed_at": 11.0,
        }
        self.harness.observe_payload(first)
        self.harness.observe_payload(second)
        capsule = self.harness.before_model(
            {
                "namespace": "run-42",
                "relation_type": "task-phase",
                "seeds": ["checkpoint-0"],
                "root": "checkpoint-0",
                "as_of": 12.0,
                "tags": ["project-x"],
                "max_hops": 2,
                "max_nodes": 8,
            }
        )
        self.assertEqual(
            [belief["node"] for belief in capsule["beliefs"]],
            ["checkpoint-0", "checkpoint-1", "checkpoint-2"],
        )
        self.assertEqual(capsule["retrieval"]["mode"], "neighborhood")
        self.assertEqual(
            capsule["retrieval"]["seed_nodes"], ["checkpoint-0"]
        )
        self.assertEqual(capsule["retrieval"]["hops_explored"], 2)
        self.assertFalse(capsule["retrieval"]["truncated"])

    def test_query_requires_exactly_one_node_selection_mode(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.harness.before_model(
                {
                    **self.query(),
                    "seeds": ["checkpoint-0"],
                }
            )
        payload = dict(self.query())
        payload.pop("nodes")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.harness.before_model(payload)


if __name__ == "__main__":
    unittest.main()
