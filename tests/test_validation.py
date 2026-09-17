import unittest

from posterior_memory_harness import (
    INPUT_SCHEMA_VERSION,
    MemoryQuery,
    PayloadValidationError,
    PosteriorMemoryHarness,
    cyclic_law,
    validate_outcome_payload,
)
from posterior_memory_harness.validation import (
    MAX_PROVENANCE_BYTES,
    MAX_QUERY_NODES,
    MAX_RELATION_STATES,
)


class PayloadValidationTests(unittest.TestCase):
    def setUp(self):
        self.harness = PosteriorMemoryHarness().register_relation(
            "phase", cyclic_law(2)
        )
        self.observation = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "observation_id": "obs-1",
            "namespace": "run",
            "relation_type": "phase",
            "source": "a",
            "target": "b",
            "posterior": {"0": 0.2, "1": 0.8},
            "prior": {"0": 0.5, "1": 0.5},
            "evidence_family": "event-1",
        }
        self.query = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "namespace": "run",
            "relation_type": "phase",
            "nodes": ["a", "b"],
            "root": "a",
        }

    def test_versioned_payloads_are_accepted(self):
        self.assertEqual(self.harness.observe_payload(self.observation), "obs-1")
        result = self.harness.query_payload(self.query)
        self.assertEqual(result["used_observation_ids"], ["obs-1"])

    def test_v2_lineage_and_retrieval_policy_are_strict(self):
        self.harness.observe_payload(
            {
                **self.observation,
                "observation_id": "upstream-1",
                "source_event_id": "tool-event-1",
                "lineage_root_id": "root-event-1",
                "derived_from": [],
            }
        )
        payload = {
            **self.observation,
            "source_event_id": "tool-event-1",
            "lineage_root_id": "root-event-1",
            "derived_from": ["upstream-1"],
            "encoder_revision": "encoder:v3",
            "calibration_revision": "temperature:v2",
        }
        self.assertEqual(self.harness.observe_payload(payload), "obs-1")
        stored = self.harness.store.retrieve(
            MemoryQuery(
                "run",
                "phase",
                ("a", "b"),
                "a",
                1.0e12,
            )
        )[0]
        self.assertEqual(stored.lineage_root_id, "root-event-1")
        result = self.harness.query_payload(
            {
                **self.query,
                "retrieval_policy": "cycle_aware",
            }
        )
        self.assertEqual(result["retrieval"]["policy"], "cycle_aware")
        with self.assertRaisesRegex(
            PayloadValidationError,
            "retrieval_policy",
        ):
            self.harness.query_payload(
                {
                    **self.query,
                    "retrieval_policy": "unknown",
                }
            )

    def test_legacy_v1_rejects_v2_only_fields_but_remains_accepted(self):
        legacy = {
            **self.observation,
            "schema_version": 1,
        }
        self.assertEqual(self.harness.observe_payload(legacy), "obs-1")
        with self.assertRaisesRegex(PayloadValidationError, "unknown fields"):
            self.harness.query_payload(
                {
                    **self.query,
                    "schema_version": 1,
                    "retrieval_policy": "coverage",
                }
            )

    def test_unknown_fields_and_future_versions_are_rejected(self):
        with self.assertRaisesRegex(PayloadValidationError, "unknown fields"):
            self.harness.observe_payload(
                {**self.observation, "independant_evidence": True}
            )
        with self.assertRaisesRegex(PayloadValidationError, "unsupported"):
            self.harness.query_payload(
                {**self.query, "schema_version": INPUT_SCHEMA_VERSION + 1}
            )

    def test_string_booleans_and_boolean_probabilities_are_rejected(self):
        with self.assertRaisesRegex(PayloadValidationError, "must be a boolean"):
            self.harness.observe_payload(
                {**self.observation, "independent_evidence": "false"}
            )
        bad = dict(self.observation)
        bad["posterior"] = [True, 0.5]
        with self.assertRaisesRegex(PayloadValidationError, "must be a number"):
            self.harness.observe_payload(bad)

    def test_array_shape_and_query_resource_limits_are_enforced(self):
        with self.assertRaisesRegex(PayloadValidationError, "tags must be an array"):
            self.harness.observe_payload({**self.observation, "tags": "unsafe"})
        oversized = [f"node-{index}" for index in range(MAX_QUERY_NODES + 1)]
        with self.assertRaisesRegex(PayloadValidationError, "between 1 and"):
            self.harness.query_payload(
                {
                    **self.query,
                    "nodes": oversized,
                    "root": oversized[0],
                }
            )
        with self.assertRaisesRegex(
            PayloadValidationError, "only valid with seeds"
        ):
            self.harness.query_payload({**self.query, "max_hops": 2})
        with self.assertRaisesRegex(PayloadValidationError, "at most"):
            self.harness.observe_payload(
                {
                    **self.observation,
                    "posterior": [1.0] * (MAX_RELATION_STATES + 1),
                }
            )
        with self.assertRaisesRegex(PayloadValidationError, "outside the query"):
            self.harness.query_payload(
                {
                    **self.query,
                    "node_priors": {"not-selected": [0.5, 0.5]},
                }
            )

    def test_provenance_has_a_serialized_size_limit(self):
        oversized = "x" * (MAX_PROVENANCE_BYTES + 1)
        with self.assertRaisesRegex(PayloadValidationError, "provenance exceeds"):
            self.harness.observe_payload(
                {**self.observation, "provenance": {"text": oversized}}
            )

    def test_required_evidence_family_cannot_be_synthesized(self):
        payload = dict(self.observation)
        payload.pop("evidence_family")
        with self.assertRaisesRegex(PayloadValidationError, "evidence_family"):
            self.harness.observe_payload(payload)

    def test_outcome_contract_is_strict_and_versioned(self):
        valid = validate_outcome_payload(
            {
                "schema_version": 1,
                "observation_id": "obs-1",
                "true_relation": "1",
                "outcome_event_id": "event-1",
                "labeled_at": 2,
            }
        )
        self.assertEqual(valid["labeled_at"], 2.0)
        with self.assertRaisesRegex(PayloadValidationError, "unknown fields"):
            validate_outcome_payload(
                {
                    **valid,
                    "unexpected": True,
                }
            )
        with self.assertRaisesRegex(PayloadValidationError, "unsupported"):
            validate_outcome_payload(
                {
                    **valid,
                    "schema_version": INPUT_SCHEMA_VERSION + 1,
                }
            )
        with self.assertRaisesRegex(PayloadValidationError, "finite"):
            validate_outcome_payload(
                {
                    **valid,
                    "labeled_at": float("nan"),
                }
            )
        with self.assertRaisesRegex(PayloadValidationError, "must be a number"):
            validate_outcome_payload(
                {
                    **valid,
                    "labeled_at": True,
                }
            )
        with self.assertRaisesRegex(PayloadValidationError, "exceeds"):
            validate_outcome_payload(
                {
                    **valid,
                    "outcome_event_id": "x" * 513,
                }
            )


if __name__ == "__main__":
    unittest.main()
