import tempfile
import unittest
from pathlib import Path

from posterior_memory_harness import (
    MemoryNeighborhoodQuery,
    MemoryObservation,
    MemoryQuery,
)
from posterior_memory_harness.store import (
    DuplicateObservationError,
    InMemoryMemoryStore,
    ObservationNotFoundError,
    SQLiteMemoryStore,
)


def observation(
    identifier: str,
    *,
    observed_at: float = 1.0,
    valid_from=None,
    valid_until=None,
    tags=(),
    source="a",
    target="b",
    independent=True,
    family=None,
):
    return MemoryObservation(
        observation_id=identifier,
        namespace="agent",
        relation_type="phase",
        source=source,
        target=target,
        posterior=(0.8, 0.2),
        prior=(0.5, 0.5),
        evidence_family=family or identifier,
        observed_at=observed_at,
        valid_from=valid_from,
        valid_until=valid_until,
        tags=tags,
        independent_evidence=independent,
    )


def query(*, as_of=5.0, tags=(), maximum=512):
    return MemoryQuery(
        namespace="agent",
        relation_type="phase",
        nodes=("a", "b"),
        root="a",
        as_of=as_of,
        tags=tags,
        max_observations=maximum,
    )


class StoreContract:
    def make_store(self):
        raise NotImplementedError

    def test_duplicate_and_missing_revoke(self):
        store = self.make_store()
        store.append(observation("one"))
        with self.assertRaises(DuplicateObservationError):
            store.append(observation("one"))
        store.revoke("one", 2.0)
        with self.assertRaises(ObservationNotFoundError):
            store.revoke("one", 3.0)
        with self.assertRaises(ObservationNotFoundError):
            store.revoke("missing", 2.0)

    def test_time_validity_and_time_travel_revoke(self):
        store = self.make_store()
        store.append(observation("window", valid_from=2.0, valid_until=8.0))
        store.revoke("window", 6.0)
        self.assertEqual(len(store.retrieve(query(as_of=5.0))), 1)
        self.assertEqual(len(store.retrieve(query(as_of=7.0))), 0)
        self.assertEqual(len(store.retrieve(query(as_of=9.0))), 0)

    def test_as_of_query_cannot_see_future_observation(self):
        store = self.make_store()
        store.append(observation("future", observed_at=100.0))
        self.assertEqual(store.retrieve(query(as_of=50.0)), [])
        self.assertEqual(
            [value.observation_id for value in store.retrieve(query(as_of=100.0))],
            ["future"],
        )

    def test_tag_filter_precedes_limit(self):
        store = self.make_store()
        store.append(observation("wanted", observed_at=1.0, tags=("project-x",)))
        for index in range(20):
            store.append(observation(f"new-{index}", observed_at=100.0 + index))
        found = store.retrieve(query(tags=("project-x",), maximum=1))
        self.assertEqual([value.observation_id for value in found], ["wanted"])

    def test_revoke_cannot_predate_observation(self):
        store = self.make_store()
        store.append(observation("late", observed_at=100.0))
        with self.assertRaisesRegex(ValueError, "must not precede"):
            store.revoke("late", 99.0)
        self.assertEqual(
            [value.observation_id for value in store.retrieve(query(as_of=101.0))],
            ["late"],
        )

    def test_bounded_neighborhood_discovers_only_reachable_active_nodes(self):
        store = self.make_store()
        store.append(
            observation("ab", source="a", target="b", tags=("project-x",))
        )
        store.append(
            observation(
                "bc",
                source="b",
                target="c",
                observed_at=2.0,
                tags=("project-x",),
            )
        )
        store.append(
            observation(
                "cd",
                source="c",
                target="d",
                observed_at=3.0,
                tags=("project-x",),
            )
        )
        store.append(
            observation(
                "wrong-tag",
                source="a",
                target="secret",
                tags=("other-project",),
            )
        )
        store.append(
            observation(
                "future-edge",
                source="a",
                target="future",
                observed_at=100.0,
                tags=("project-x",),
            )
        )
        store.append(
            observation(
                "derived",
                source="a",
                target="summary-only",
                tags=("project-x",),
                independent=False,
            )
        )
        neighborhood = store.retrieve_neighborhood(
            MemoryNeighborhoodQuery(
                namespace="agent",
                relation_type="phase",
                seeds=("a",),
                root="a",
                as_of=10.0,
                tags=("project-x",),
                max_hops=2,
                max_nodes=10,
            )
        )
        self.assertEqual(neighborhood.nodes, ("a", "b", "c"))
        self.assertEqual(
            [value.observation_id for value in neighborhood.observations],
            ["ab", "bc"],
        )
        self.assertEqual(neighborhood.hops_explored, 2)
        self.assertFalse(neighborhood.node_limit_hit)
        self.assertFalse(neighborhood.observation_limit_hit)

    def test_neighborhood_reports_node_and_observation_truncation(self):
        store = self.make_store()
        store.append(observation("ab", source="a", target="b"))
        store.append(observation("bc", source="b", target="c", observed_at=2.0))

        node_limited = store.retrieve_neighborhood(
            MemoryNeighborhoodQuery(
                "agent",
                "phase",
                ("a",),
                "a",
                10.0,
                max_hops=3,
                max_nodes=2,
            )
        )
        self.assertEqual(node_limited.nodes, ("a", "b"))
        self.assertTrue(node_limited.node_limit_hit)

        observation_limited = store.retrieve_neighborhood(
            MemoryNeighborhoodQuery(
                "agent",
                "phase",
                ("a",),
                "a",
                10.0,
                max_hops=3,
                max_nodes=10,
                max_observations=1,
            )
        )
        self.assertEqual(
            [value.observation_id for value in observation_limited.observations],
            ["ab"],
        )
        self.assertTrue(observation_limited.observation_limit_hit)

    def test_neighborhood_uses_latest_observation_per_evidence_family(self):
        store = self.make_store()
        store.append(
            observation(
                "older",
                source="a",
                target="old-node",
                observed_at=1.0,
                family="raw-event",
            )
        )
        store.append(
            observation(
                "newer",
                source="a",
                target="new-node",
                observed_at=2.0,
                family="raw-event",
            )
        )
        neighborhood = store.retrieve_neighborhood(
            MemoryNeighborhoodQuery(
                "agent", "phase", ("a",), "a", 10.0, max_hops=1
            )
        )
        self.assertEqual(neighborhood.nodes, ("a", "new-node"))
        self.assertEqual(
            [value.observation_id for value in neighborhood.observations],
            ["newer"],
        )


class InMemoryStoreTests(StoreContract, unittest.TestCase):
    def make_store(self):
        return InMemoryMemoryStore()


class SQLiteStoreTests(StoreContract, unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary.cleanup()

    def make_store(self):
        return SQLiteMemoryStore(Path(self.temporary.name) / "memory.sqlite")

    def test_persists_across_instances(self):
        path = Path(self.temporary.name) / "persistent.sqlite"
        SQLiteMemoryStore(path).append(observation("persisted"))
        restored = SQLiteMemoryStore(path).retrieve(query())
        self.assertEqual(restored[0].observation_id, "persisted")


if __name__ == "__main__":
    unittest.main()
