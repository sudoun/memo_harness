from contextlib import closing
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from posterior_memory_harness import (
    InMemoryMemoryStore,
    MemoryNeighborhoodQuery,
    MemoryObservation,
    MemoryQuery,
    PosteriorMemoryHarness,
    SQLiteMemoryStore,
    cyclic_law,
)
from posterior_memory_harness.retrieval import (
    deduplicate_lineages,
    select_budgeted_observations,
)


def observation(
    identifier: str,
    source: str,
    target: str,
    posterior: tuple[float, float],
    observed_at: float,
    *,
    evidence_family: str | None = None,
    lineage_root_id: str | None = None,
    source_event_id: str | None = None,
    derived_from: tuple[str, ...] = (),
) -> MemoryObservation:
    return MemoryObservation(
        observation_id=identifier,
        namespace="agent",
        relation_type="phase",
        source=source,
        target=target,
        posterior=posterior,
        prior=(0.5, 0.5),
        evidence_family=evidence_family or identifier,
        observed_at=observed_at,
        lineage_root_id=lineage_root_id,
        source_event_id=source_event_id,
        derived_from=derived_from,
        encoder_revision="test-encoder:v1",
        calibration_revision="temperature:test-v1",
    )


def triangle_with_recent_noise() -> list[MemoryObservation]:
    return [
        observation("old-ab", "a", "b", (0.1, 0.9), 1.0),
        observation("old-bc", "b", "c", (0.9, 0.1), 2.0),
        observation("old-ac", "a", "c", (0.1, 0.9), 3.0),
        observation("new-ab-1", "a", "b", (0.5, 0.5), 10.0),
        observation("new-ab-2", "a", "b", (0.5, 0.5), 11.0),
        observation("new-ab-3", "a", "b", (0.5, 0.5), 12.0),
    ]


class StructuralRetrievalTests(unittest.TestCase):
    def test_lineage_root_deduplicates_cross_family_echoes(self):
        raw = observation(
            "raw",
            "a",
            "b",
            (0.8, 0.2),
            1.0,
            evidence_family="tool-parser",
            source_event_id="event-7",
            lineage_root_id="event-7",
        )
        echo = observation(
            "echo",
            "a",
            "b",
            (0.7, 0.3),
            2.0,
            evidence_family="model-echo",
            source_event_id="model-turn-8",
            lineage_root_id="event-7",
            derived_from=("raw",),
        )
        self.assertEqual(
            [item.observation_id for item in deduplicate_lineages([raw, echo])],
            ["echo"],
        )

    def test_cycle_aware_uses_same_budget_but_preserves_useful_cycle(self):
        values = triangle_with_recent_noise()
        common = {
            "nodes": ("a", "b", "c"),
            "root": "a",
            "limit": 3,
        }
        recent = select_budgeted_observations(
            values,
            policy="recent",
            **common,
        )
        coverage = select_budgeted_observations(
            values,
            policy="coverage",
            **common,
        )
        cycle = select_budgeted_observations(
            values,
            policy="cycle_aware",
            **common,
        )
        self.assertEqual(len(recent), len(coverage))
        self.assertEqual(len(coverage), len(cycle))
        self.assertEqual(
            {item.observation_id for item in recent},
            {"new-ab-1", "new-ab-2", "new-ab-3"},
        )
        self.assertEqual(
            {item.observation_id for item in coverage},
            {"old-ac", "old-bc", "new-ab-3"},
        )
        self.assertEqual(
            {item.observation_id for item in cycle},
            {"old-ab", "old-bc", "old-ac"},
        )

    def test_selection_is_invariant_to_candidate_input_order(self):
        values = triangle_with_recent_noise()
        expected = {}
        for policy in ("recent", "coverage", "cycle_aware"):
            expected[policy] = tuple(
                item.observation_id
                for item in select_budgeted_observations(
                    values,
                    nodes=("a", "b", "c"),
                    root="a",
                    limit=3,
                    policy=policy,
                )
            )
        random.Random(20260725).shuffle(values)
        for policy, identifiers in expected.items():
            actual = tuple(
                item.observation_id
                for item in select_budgeted_observations(
                    values,
                    nodes=("a", "b", "c"),
                    root="a",
                    limit=3,
                    policy=policy,
                )
            )
            self.assertEqual(actual, identifiers)

    def test_cycle_aware_changes_decision_under_matched_observation_budget(self):
        for store in (InMemoryMemoryStore(),):
            harness = PosteriorMemoryHarness(store).register_relation(
                "phase",
                cyclic_law(2),
            )
            for value in triangle_with_recent_noise():
                harness.observe(value)
            recent = harness.query(
                MemoryQuery(
                    "agent",
                    "phase",
                    ("a", "b", "c"),
                    "a",
                    20.0,
                    max_observations=3,
                    retrieval_policy="recent",
                )
            )
            cycle = harness.query(
                MemoryQuery(
                    "agent",
                    "phase",
                    ("a", "b", "c"),
                    "a",
                    20.0,
                    max_observations=3,
                    retrieval_policy="cycle_aware",
                )
            )
            recent_states = {
                belief.node: belief.candidates[0].state
                for belief in recent.beliefs
            }
            cycle_states = {
                belief.node: belief.candidates[0].state
                for belief in cycle.beliefs
            }
            self.assertEqual(recent_states, {"a": "0", "b": "0", "c": "0"})
            self.assertEqual(cycle_states, {"a": "0", "b": "1", "c": "1"})
            self.assertEqual(
                recent.retrieval.selected_observation_count,
                cycle.retrieval.selected_observation_count,
            )
            self.assertEqual(cycle.retrieval.cycle_rank, 1)
            self.assertEqual(cycle.retrieval.covered_node_fraction, 1.0)
            self.assertGreater(cycle.retrieval.cycle_information_score, 0.0)

    def test_exact_lineage_echo_does_not_change_inferred_beliefs(self):
        harness = PosteriorMemoryHarness().register_relation(
            "phase",
            cyclic_law(2),
        )
        raw = observation(
            "raw",
            "a",
            "b",
            (0.1, 0.9),
            1.0,
            lineage_root_id="event-1",
        )
        harness.observe(raw)
        query = MemoryQuery(
            "agent",
            "phase",
            ("a", "b"),
            "a",
            10.0,
        )
        before = harness.query(query)
        harness.observe(
            observation(
                "echo",
                "a",
                "b",
                (0.1, 0.9),
                2.0,
                evidence_family="model-echo",
                lineage_root_id="event-1",
                derived_from=("raw",),
            )
        )
        after = harness.query(query)
        self.assertEqual(
            [belief.to_dict() for belief in before.beliefs],
            [belief.to_dict() for belief in after.beliefs],
        )
        self.assertEqual(after.used_observation_ids, ("echo",))
        self.assertEqual(after.retrieval.independent_lineage_count, 1)

    def test_lineage_parent_must_exist_and_independent_echo_retains_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            stores = (
                InMemoryMemoryStore(),
                SQLiteMemoryStore(Path(directory) / "memory.sqlite"),
            )
            for store in stores:
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    store.append(
                        observation(
                            "orphan",
                            "a",
                            "b",
                            (0.2, 0.8),
                            2.0,
                            lineage_root_id="event-1",
                            derived_from=("missing",),
                        )
                    )
                parent = observation(
                    "parent",
                    "a",
                    "b",
                    (0.2, 0.8),
                    1.0,
                    lineage_root_id="event-1",
                )
                store.append(parent)
                with self.assertRaisesRegex(ValueError, "lineage root"):
                    store.append(
                        observation(
                            "wrong-root",
                            "a",
                            "b",
                            (0.2, 0.8),
                            2.0,
                            lineage_root_id="event-2",
                            derived_from=("parent",),
                        )
                    )
                with self.assertRaisesRegex(ValueError, "endpoints"):
                    store.append(
                        observation(
                            "wrong-edge",
                            "a",
                            "c",
                            (0.2, 0.8),
                            2.0,
                            lineage_root_id="event-1",
                            derived_from=("parent",),
                        )
                    )

    def test_sqlite_round_trips_lineage_and_structural_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory.sqlite")
            harness = PosteriorMemoryHarness(store).register_relation(
                "phase",
                cyclic_law(2),
            )
            for value in triangle_with_recent_noise():
                harness.observe(value)
            harness.observe(
                observation(
                    "upstream",
                    "a",
                    "b",
                    (0.8, 0.2),
                    0.4,
                    source_event_id="source-event",
                    lineage_root_id="source-root",
                )
            )
            lineage = observation(
                "lineage",
                "a",
                "b",
                (0.8, 0.2),
                0.5,
                source_event_id="source-event",
                lineage_root_id="source-root",
                derived_from=("upstream",),
            )
            harness.observe(lineage)
            restored = {
                item.observation_id: item
                for item in store.retrieve(
                    MemoryQuery(
                        "agent",
                        "phase",
                        ("a", "b", "c"),
                        "a",
                        20.0,
                        max_observations=32,
                    )
                )
            }
            self.assertEqual(restored["lineage"].source_event_id, "source-event")
            self.assertEqual(restored["lineage"].lineage_root_id, "source-root")
            self.assertEqual(restored["lineage"].derived_from, ("upstream",))
            capsule = harness.query(
                MemoryQuery(
                    "agent",
                    "phase",
                    ("a", "b", "c"),
                    "a",
                    20.0,
                    max_observations=3,
                    retrieval_policy="cycle_aware",
                )
            )
            payload = capsule.to_dict()["retrieval"]
            self.assertEqual(payload["policy"], "cycle_aware")
            self.assertEqual(payload["selected_observation_count"], 3)
            self.assertEqual(payload["independent_lineage_count"], 3)
            self.assertEqual(payload["connected_components"], 1)
            self.assertEqual(payload["cycle_rank"], 1)
            self.assertGreater(payload["cycle_information_score"], 0.0)

    def test_in_memory_and_sqlite_select_identical_budgeted_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            stores = (
                InMemoryMemoryStore(),
                SQLiteMemoryStore(Path(directory) / "memory.sqlite"),
            )
            harnesses = [
                PosteriorMemoryHarness(store).register_relation(
                    "phase",
                    cyclic_law(2),
                )
                for store in stores
            ]
            for value in triangle_with_recent_noise():
                for harness in harnesses:
                    harness.observe(value)
            for policy in ("recent", "coverage", "cycle_aware"):
                query = MemoryQuery(
                    "agent",
                    "phase",
                    ("a", "b", "c"),
                    "a",
                    20.0,
                    max_observations=3,
                    retrieval_policy=policy,
                )
                identifiers = [
                    harness.query(query).used_observation_ids
                    for harness in harnesses
                ]
                self.assertEqual(identifiers[0], identifiers[1])

    def test_integrity_verifier_detects_broken_lineage_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            store = SQLiteMemoryStore(path)
            harness = PosteriorMemoryHarness(store).register_relation(
                "phase",
                cyclic_law(2),
            )
            harness.observe(
                observation(
                    "parent",
                    "a",
                    "b",
                    (0.2, 0.8),
                    1.0,
                    lineage_root_id="event-1",
                )
            )
            harness.observe(
                observation(
                    "echo",
                    "a",
                    "b",
                    (0.2, 0.8),
                    2.0,
                    lineage_root_id="event-1",
                    derived_from=("parent",),
                )
            )
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    """
                    UPDATE observations
                    SET derived_from_json = '["missing"]'
                    WHERE observation_id = 'echo'
                    """
                )
            report = store.verify_integrity()
            self.assertFalse(report["ok"])
            self.assertTrue(
                any(
                    "parent 'missing' is missing" in error
                    for error in report["errors"]
                )
            )

    def test_cycle_aware_neighborhood_discovers_before_spending_factor_budget(self):
        store = InMemoryMemoryStore()
        for value in triangle_with_recent_noise():
            store.append(value)
        recent = store.retrieve_neighborhood(
            MemoryNeighborhoodQuery(
                "agent",
                "phase",
                ("a",),
                "a",
                20.0,
                max_hops=2,
                max_nodes=3,
                max_observations=3,
                retrieval_policy="recent",
            )
        )
        cycle = store.retrieve_neighborhood(
            MemoryNeighborhoodQuery(
                "agent",
                "phase",
                ("a",),
                "a",
                20.0,
                max_hops=2,
                max_nodes=3,
                max_observations=3,
                retrieval_policy="cycle_aware",
            )
        )
        self.assertEqual(set(recent.nodes), {"a", "b"})
        self.assertEqual(set(cycle.nodes), {"a", "b", "c"})
        self.assertEqual(
            {item.observation_id for item in cycle.observations},
            {"old-ab", "old-bc", "old-ac"},
        )


if __name__ == "__main__":
    unittest.main()
