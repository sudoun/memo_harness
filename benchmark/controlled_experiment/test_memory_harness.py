#!/usr/bin/env python3

import unittest

import numpy as np

from memory_harness import MemoryAtom, PosteriorGlueMemoryHarness, s3_relation_law


class MemoryHarnessTest(unittest.TestCase):
    def setUp(self):
        self.law = s3_relation_law()

    def atom(self, observation_id, source, target, relation, family=None):
        posterior = np.full(self.law.size, 0.01)
        posterior[relation] = 0.95
        return MemoryAtom(
            observation_id=observation_id,
            source=source,
            target=target,
            posterior=posterior,
            prior=np.full(self.law.size, 1 / self.law.size),
            evidence_family=family or observation_id,
        )

    def test_s3_is_associative_and_noncommutative(self):
        table = self.law.table
        for left in range(6):
            for middle in range(6):
                for right in range(6):
                    self.assertEqual(
                        table[table[left, middle], right],
                        table[left, table[middle, right]],
                    )
        self.assertTrue(
            any(
                table[left, right] != table[right, left]
                for left in range(6)
                for right in range(6)
            )
        )

    def test_exact_chain_recovers_composed_state(self):
        harness = PosteriorGlueMemoryHarness(self.law)
        first, second = 2, 4
        harness.observe(self.atom("a", "start", "mid", first))
        harness.observe(self.atom("b", "mid", "end", second))
        result = harness.query(("start", "mid", "end"), "start")
        expected = self.law.compose(first, second)
        self.assertEqual(int(np.argmax(result.marginals[2])), expected)

    def test_evidence_family_is_counted_once(self):
        base = PosteriorGlueMemoryHarness(self.law)
        duplicate = PosteriorGlueMemoryHarness(self.law)
        atom = self.atom("original", "start", "end", 3, family="tool-output-1")
        base.observe(atom)
        duplicate.observe(atom)
        duplicate.observe(
            self.atom("summary", "start", "end", 3, family="tool-output-1")
        )
        first = base.query(("start", "end"), "start").marginals
        second = duplicate.query(("start", "end"), "start").marginals
        self.assertTrue(np.allclose(first, second))

    def test_prior_correction_recovers_observation_likelihood(self):
        likelihood = np.asarray([1.0, 4.0, 2.0, 3.0, 5.0, 1.0])
        prior = np.asarray([0.05, 0.10, 0.20, 0.25, 0.30, 0.10])
        posterior = likelihood * prior
        atom = MemoryAtom(
            observation_id="prior-test",
            source="start",
            target="end",
            posterior=posterior,
            prior=prior,
            evidence_family="prior-test",
        )
        self.assertTrue(
            np.allclose(atom.likelihood(), likelihood / likelihood.sum())
        )

    def test_capsule_never_promotes_memory_text_to_instructions(self):
        harness = PosteriorGlueMemoryHarness(self.law)
        harness.observe(self.atom("a", "start", "end", 1))
        capsule = harness.before_model({"nodes": ("start", "end"), "root": "start"})
        self.assertEqual(capsule["instructions"], [])
        self.assertEqual(capsule["used_observation_ids"], ["a"])


if __name__ == "__main__":
    unittest.main()
