import unittest

import numpy as np

from posterior_memory_harness import MemoryObservation, MemoryQuery, cyclic_law
from posterior_memory_harness.inference import (
    InferenceConfig,
    deduplicate_observations,
    infer,
)


def edge(identifier, source, target, posterior, family=None, observed_at=1.0):
    return MemoryObservation(
        observation_id=identifier,
        namespace="n",
        relation_type="c3",
        source=source,
        target=target,
        posterior=tuple(posterior),
        prior=(1 / 3, 1 / 3, 1 / 3),
        evidence_family=family or identifier,
        observed_at=observed_at,
    )


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.law = cyclic_law(3)
        self.config = InferenceConfig(
            max_exact_assignments=10_000,
            beam_size=100,
            point_mass_error=1e-6,
        )
        self.observations = [
            edge("ab", "a", "b", (0.01, 0.98, 0.01)),
            edge("bc", "b", "c", (0.01, 0.01, 0.98)),
            edge("ac", "a", "c", (0.98, 0.01, 0.01)),
        ]

    def query(self, backend="exact", hard=False):
        return MemoryQuery(
            namespace="n",
            relation_type="c3",
            nodes=("a", "b", "c"),
            root="a",
            as_of=5.0,
            hard_top1=hard,
            inference=backend,
        )

    def test_exact_cycle_recovers_composed_state(self):
        output = infer(self.law, self.query(), self.observations, self.config)
        self.assertEqual(output.map_assignment.tolist(), [0, 1, 0])
        self.assertGreater(output.marginals[1, 1], 0.99)
        self.assertGreater(output.marginals[2, 0], 0.99)

    def test_prior_correction_removes_prior_only_information(self):
        observation = MemoryObservation(
            observation_id="prior",
            namespace="n",
            relation_type="c2",
            source="a",
            target="b",
            posterior=(0.9, 0.1),
            prior=(0.9, 0.1),
            evidence_family="prior",
        )
        query = MemoryQuery("n", "c2", ("a", "b"), "a", 1.0, inference="exact")
        output = infer(cyclic_law(2), query, [observation], self.config)
        np.testing.assert_allclose(output.marginals[1], (0.5, 0.5), atol=1e-10)

    def test_duplicate_evidence_family_uses_latest_only(self):
        older = edge(
            "older", "a", "b", (0.01, 0.98, 0.01), "same-raw-event", 1.0
        )
        newer = edge(
            "newer", "a", "b", (0.01, 0.01, 0.98), "same-raw-event", 2.0
        )
        selected = deduplicate_observations([newer, older])
        self.assertEqual([value.observation_id for value in selected], ["newer"])

    def test_derived_compaction_is_not_counted_as_new_evidence(self):
        derived = MemoryObservation(
            **{
                **self.observations[0].to_dict(),
                "observation_id": "summary",
                "posterior": (0.01, 0.98, 0.01),
                "prior": (1 / 3, 1 / 3, 1 / 3),
                "evidence_family": "summary",
                "independent_evidence": False,
            }
        )
        selected = deduplicate_observations([self.observations[0], derived])
        self.assertEqual([value.observation_id for value in selected], ["ab"])

    def test_full_width_beam_matches_exact(self):
        exact = infer(self.law, self.query("exact"), self.observations, self.config)
        beam = infer(self.law, self.query("beam"), self.observations, self.config)
        np.testing.assert_allclose(beam.marginals, exact.marginals, atol=1e-12)
        self.assertEqual(beam.map_assignment.tolist(), exact.map_assignment.tolist())
        self.assertAlmostEqual(beam.log_evidence, exact.log_evidence)
        self.assertEqual(exact.diagnostics.mode, "exact")
        self.assertTrue(exact.diagnostics.stable)
        self.assertEqual(beam.diagnostics.mode, "dual-width-beam")
        self.assertTrue(beam.diagnostics.stable)
        self.assertEqual(beam.diagnostics.base_beam_size, 100)
        self.assertEqual(beam.diagnostics.comparison_beam_size, 200)

    def test_dual_width_beam_exposes_truncation_instability(self):
        law = cyclic_law(2)
        observation = MemoryObservation(
            observation_id="ab",
            namespace="n",
            relation_type="c2",
            source="a",
            target="b",
            posterior=(0.6, 0.4),
            prior=(0.5, 0.5),
            evidence_family="ab",
        )
        query = MemoryQuery(
            "n",
            "c2",
            ("a", "b"),
            "a",
            1.0,
            inference="beam",
        )
        output = infer(
            law,
            query,
            [observation],
            InferenceConfig(
                beam_size=1,
                beam_stability_multiplier=2,
                beam_marginal_tv_tolerance=0.01,
                beam_entropy_tolerance=0.01,
            ),
        )
        np.testing.assert_allclose(output.marginals[1], (0.6, 0.4))
        self.assertEqual(output.backend, "beam-2")
        self.assertFalse(output.diagnostics.stable)
        self.assertTrue(output.diagnostics.map_agreement)
        self.assertAlmostEqual(output.diagnostics.max_marginal_tv, 0.4)
        self.assertGreater(output.diagnostics.max_entropy_delta, 0.6)

    def test_beam_stability_check_can_be_disabled_explicitly(self):
        output = infer(
            self.law,
            self.query("beam"),
            self.observations,
            InferenceConfig(
                max_exact_assignments=10_000,
                beam_size=100,
                beam_stability_check=False,
            ),
        )
        self.assertEqual(output.backend, "beam-100")
        self.assertEqual(output.diagnostics.mode, "beam-unchecked")
        self.assertIsNone(output.diagnostics.stable)

    def test_beam_scores_root_self_factor(self):
        root_factor = edge("aa", "a", "a", (0.01, 0.98, 0.01))
        exact = infer(self.law, self.query("exact"), [root_factor], self.config)
        beam = infer(self.law, self.query("beam"), [root_factor], self.config)
        self.assertAlmostEqual(beam.log_evidence, exact.log_evidence)


if __name__ == "__main__":
    unittest.main()
