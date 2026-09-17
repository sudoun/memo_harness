import unittest

import numpy as np

from posterior_memory_harness import (
    MemoryObservation,
    MemoryNeighborhoodQuery,
    MemoryQuery,
    RelationLawRegistry,
    cyclic_law,
    s3_law,
)
from posterior_memory_harness.laws import FiniteRelationLaw


class LawAndModelTests(unittest.TestCase):
    def test_s3_is_noncommutative_and_valid(self):
        law = s3_law()
        pairs = [
            (left, right)
            for left in range(law.size)
            for right in range(law.size)
            if law.compose(left, right) != law.compose(right, left)
        ]
        self.assertTrue(pairs)
        for value in range(law.size):
            self.assertEqual(law.relative(value, value), law.identity)

    def test_custom_invalid_law_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "associative"):
            FiniteRelationLaw(
                name="bad",
                labels=("e", "a", "b"),
                table=np.asarray([[0, 1, 2], [1, 0, 1], [2, 2, 0]]),
                inverse=np.asarray([0, 1, 2]),
            )

    def test_probability_validation_and_prior_correction(self):
        observation = MemoryObservation(
            observation_id="o",
            namespace="n",
            relation_type="r",
            source="a",
            target="b",
            posterior=(0.9, 0.1),
            prior=(0.9, 0.1),
            evidence_family="raw-event-1",
        )
        self.assertAlmostEqual(observation.likelihood()[0], 0.5)
        self.assertAlmostEqual(observation.likelihood()[1], 0.5)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MemoryObservation(
                observation_id="bad",
                namespace="n",
                relation_type="r",
                source="a",
                target="b",
                posterior=(1.1, -0.1),
                prior=(0.5, 0.5),
                evidence_family="bad",
            )

    def test_query_contract(self):
        query = MemoryQuery(
            namespace="n",
            relation_type="r",
            nodes=("root", "x"),
            root="root",
            as_of=10.0,
            tags=("b", "a", "a"),
        )
        self.assertEqual(query.tags, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "root"):
            MemoryQuery("n", "r", ("x",), "root", 10.0)

    def test_nonfinite_and_pre_observation_times_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "observed_at"):
            MemoryObservation(
                "bad-time",
                "n",
                "r",
                "a",
                "b",
                (0.5, 0.5),
                (0.5, 0.5),
                "event",
                observed_at=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "must not precede"):
            MemoryObservation(
                "bad-revoke",
                "n",
                "r",
                "a",
                "b",
                (0.5, 0.5),
                (0.5, 0.5),
                "event",
                observed_at=10.0,
                revoked_at=9.0,
            )

    def test_cyclic_one_hard_support_is_legal(self):
        self.assertEqual(cyclic_law(1).size, 1)

    def test_registered_relation_law_cannot_change_silently(self):
        registry = RelationLawRegistry()
        registry.register("phase", cyclic_law(3))
        registry.register("phase", cyclic_law(3))
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("phase", cyclic_law(4))

    def test_provenance_is_copied_and_read_only(self):
        source = {"tool": "tracker"}
        observation = MemoryObservation(
            "immutable",
            "n",
            "r",
            "a",
            "b",
            (0.5, 0.5),
            (0.5, 0.5),
            "event",
            provenance=source,
        )
        source["tool"] = "mutated"
        self.assertEqual(observation.provenance["tool"], "tracker")
        with self.assertRaises(TypeError):
            observation.provenance["tool"] = "mutated"

    def test_lineage_contract_is_canonical_and_bounded(self):
        observation = MemoryObservation(
            "echo",
            "n",
            "r",
            "a",
            "b",
            (0.5, 0.5),
            (0.5, 0.5),
            "parser-family",
            source_event_id="model-turn",
            lineage_root_id="tool-event",
            derived_from=("raw",),
        )
        self.assertEqual(observation.lineage_key, "tool-event")
        self.assertEqual(observation.derived_from, ("raw",))
        with self.assertRaisesRegex(ValueError, "at most 256"):
            MemoryObservation(
                "too-many-parents",
                "n",
                "r",
                "a",
                "b",
                (0.5, 0.5),
                (0.5, 0.5),
                "event",
                derived_from=tuple(
                    f"parent-{index}" for index in range(257)
                ),
            )

    def test_neighborhood_query_validates_root_and_budgets(self):
        query = MemoryNeighborhoodQuery(
            "n", "r", ("root",), "root", 10.0, max_hops=2, max_nodes=4
        )
        self.assertEqual(query.explicit(("root", "x")).nodes, ("root", "x"))
        with self.assertRaisesRegex(ValueError, "root"):
            MemoryNeighborhoodQuery("n", "r", ("seed",), "root", 10.0)
        with self.assertRaisesRegex(ValueError, "max_nodes"):
            MemoryNeighborhoodQuery(
                "n", "r", ("a", "b"), "a", 10.0, max_nodes=1
            )


if __name__ == "__main__":
    unittest.main()
