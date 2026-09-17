import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from posterior_memory_harness import (
    CalibrationDriftMonitor,
    MemoryObservation,
    MemoryQuery,
    MonitorConfig,
    MonitorPredictionRecord,
    PosteriorMemoryHarness,
    RelationLawRegistry,
    SQLiteMemoryStore,
    cyclic_law,
    relation_law_fingerprint,
)
from posterior_memory_harness.sqlite_schema import SchemaMigrationError
from posterior_memory_harness.store import (
    MonitorConfigurationConflictError,
)


def observation(identifier: str, observed_at: float = 1.0) -> MemoryObservation:
    return MemoryObservation(
        observation_id=identifier,
        namespace="agent",
        relation_type="phase",
        source="root",
        target=identifier,
        posterior=(0.1, 0.9),
        prior=(0.5, 0.5),
        evidence_family=f"tool:{identifier}",
        provenance={"tool": "test"},
        observed_at=observed_at,
    )


def payload_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DurableHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_same_name_noop_trigger_fails_schema_and_integrity_checks(self):
        store = SQLiteMemoryStore(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TRIGGER relation_laws_no_update")
            connection.execute(
                """
                CREATE TRIGGER relation_laws_no_update
                BEFORE UPDATE ON relation_laws
                BEGIN
                    SELECT 1;
                END
                """
            )
            connection.commit()
        with self.assertRaisesRegex(
            SchemaMigrationError, "locked v2 contract"
        ):
            store.check_schema()
        verification = store.verify_integrity()
        self.assertFalse(verification["ok"])
        self.assertFalse(verification["checks"]["schema_compatible"])

    def test_unexpected_trigger_on_locked_table_is_rejected(self):
        store = SQLiteMemoryStore(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                CREATE TRIGGER unexpected_monitor_trigger
                BEFORE INSERT ON monitor_predictions
                BEGIN
                    SELECT RAISE(ABORT, 'unexpected');
                END
                """
            )
            connection.commit()
        with self.assertRaisesRegex(
            SchemaMigrationError, "unexpected triggers"
        ):
            store.check_schema()

    def test_migration_history_is_append_only(self):
        store = SQLiteMemoryStore(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            with self.assertRaisesRegex(
                sqlite3.IntegrityError, "append-only"
            ):
                connection.execute(
                    """
                    UPDATE memory_schema_migrations
                    SET applied_at = 'forged'
                    WHERE version = 2
                    """
                )
        self.assertTrue(store.verify_integrity()["ok"])

    def test_public_record_canonicalization_cannot_poison_hash_journal(self):
        store = SQLiteMemoryStore(self.path)
        monitor = CalibrationDriftMonitor(
            MonitorConfig(
                run_id="canonical",
                reference_size=1,
                current_size=1,
                min_current_size=1,
            )
        )
        harness = PosteriorMemoryHarness(store, monitor=monitor)
        law = cyclic_law(2)
        harness.register_relation("phase", law)
        item = observation("integer-time")
        fingerprint = relation_law_fingerprint(law.to_dict())
        values = {
            "observation_id": item.observation_id,
            "run_id": "canonical",
            "namespace": item.namespace,
            "relation_type": item.relation_type,
            "source_family": "test",
            "posterior": list(item.posterior),
            "support_labels": list(law.labels),
            "law_fingerprint": fingerprint,
            "observed_at": item.observed_at,
            # Intentionally hash the non-canonical integer representation.
            "tracked_at": 2,
        }
        prediction = MonitorPredictionRecord(
            observation_id=item.observation_id,
            run_id="canonical",
            namespace=item.namespace,
            relation_type=item.relation_type,
            source_family="test",
            posterior=item.posterior,
            support_labels=law.labels,
            law_fingerprint=fingerprint,
            observed_at=item.observed_at,
            tracked_at=2,
            payload_sha256=payload_hash(values),
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            store.append_monitored(item, prediction)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM monitor_predictions"
                ).fetchone()[0],
                0,
            )

    def test_invalid_registration_does_not_poison_durable_registry(self):
        harness = PosteriorMemoryHarness(SQLiteMemoryStore(self.path))
        with self.assertRaises(ValueError):
            harness.register_relation("", cyclic_law(2))
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM relation_laws"
                ).fetchone()[0],
                0,
            )

    def test_low_level_invalid_law_snapshot_cannot_poison_registry(self):
        store = SQLiteMemoryStore(self.path)
        with self.assertRaises(ValueError):
            store.bind_relation_law(
                "", (), "X" * 64, law_payload={}
            )
        law = cyclic_law(2)
        with self.assertRaisesRegex(ValueError, "canonical law payload"):
            store.bind_relation_law(
                "phase",
                law.labels,
                "0" * 64,
                law_payload=law.to_dict(),
            )
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM relation_laws"
                ).fetchone()[0],
                0,
            )

    def test_prefilled_registry_is_bound_at_observation_boundary(self):
        registry = RelationLawRegistry()
        registry.register("phase", cyclic_law(2))
        store = SQLiteMemoryStore(self.path)
        harness = PosteriorMemoryHarness(store, registry=registry)
        harness.observe(observation("prefilled"))
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM relation_laws"
                ).fetchone()[0],
                1,
            )
        self.assertTrue(store.verify_integrity()["ok"])

    def test_unbound_low_level_observation_fails_closed(self):
        store = SQLiteMemoryStore(self.path)
        store.append(observation("raw"))
        verification = store.verify_integrity()
        self.assertFalse(verification["ok"])
        self.assertEqual(
            verification["checks"]["unbound_observation_relation_types"],
            ["phase"],
        )
        registry = RelationLawRegistry()
        registry.register("phase", cyclic_law(2))
        harness = PosteriorMemoryHarness(store, registry=registry)
        with self.assertRaisesRegex(
            MonitorConfigurationConflictError, "no database-global"
        ):
            harness.query(
                MemoryQuery(
                    namespace="agent",
                    relation_type="phase",
                    nodes=("root", "raw"),
                    root="root",
                    as_of=2.0,
                )
            )

    def test_first_legacy_binding_rejects_support_size_mismatch(self):
        store = SQLiteMemoryStore(self.path)
        store.append(observation("legacy-size-two"))
        wrong_law = cyclic_law(3)
        with self.assertRaisesRegex(
            MonitorConfigurationConflictError, "support size"
        ):
            store.bind_relation_law(
                "phase",
                wrong_law.labels,
                relation_law_fingerprint(wrong_law.to_dict()),
                law_payload=wrong_law.to_dict(),
            )
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM relation_laws"
                ).fetchone()[0],
                0,
            )

    def test_first_legacy_binding_rejects_invalid_same_size_payload(self):
        store = SQLiteMemoryStore(self.path)
        store.append(observation("legacy-invalid"))
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                UPDATE observations
                SET posterior_json = '["bad",0.4]'
                WHERE observation_id = 'legacy-invalid'
                """
            )
            connection.commit()
        law = cyclic_law(2)
        with self.assertRaisesRegex(
            MonitorConfigurationConflictError, "is invalid"
        ):
            store.bind_relation_law(
                "phase",
                law.labels,
                relation_law_fingerprint(law.to_dict()),
                law_payload=law.to_dict(),
            )
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM relation_laws"
                ).fetchone()[0],
                0,
            )

    def test_bound_law_rejects_low_level_observation_support_mismatch(self):
        store = SQLiteMemoryStore(self.path)
        law = cyclic_law(3)
        store.bind_relation_law(
            "phase",
            law.labels,
            relation_law_fingerprint(law.to_dict()),
            law_payload=law.to_dict(),
        )
        with self.assertRaisesRegex(ValueError, "support does not match"):
            store.append(observation("wrong-size"))
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0],
                0,
            )
        self.assertTrue(store.verify_integrity()["ok"])

    def test_stale_worker_replays_interleaved_sequence_without_gap(self):
        config = MonitorConfig(
            run_id="shared",
            reference_size=1,
            current_size=1,
            min_current_size=1,
        )
        first = PosteriorMemoryHarness(
            SQLiteMemoryStore(self.path),
            monitor=CalibrationDriftMonitor(config),
        ).register_relation("phase", cyclic_law(2))
        second = PosteriorMemoryHarness(
            SQLiteMemoryStore(self.path),
            monitor=CalibrationDriftMonitor(config),
        ).register_relation("phase", cyclic_law(2))

        first.observe(observation("one", 1.0))
        second.observe(observation("two", 2.0))
        first.observe(observation("three", 3.0))
        self.assertEqual(
            first.calibration_report()["relations"]["phase"]["overall"][
                "counts"
            ]["tracked"],
            3,
        )

        first.record_outcome(
            "one",
            "1",
            labeled_at=4.0,
            recorded_at=4.0,
            outcome_event_id="truth:one",
        )
        second.record_outcome(
            "two",
            "1",
            labeled_at=5.0,
            recorded_at=5.0,
            outcome_event_id="truth:two",
        )
        first.record_outcome(
            "three",
            "1",
            labeled_at=6.0,
            recorded_at=6.0,
            outcome_event_id="truth:three",
        )
        counts = first.calibration_report()["relations"]["phase"]["overall"][
            "counts"
        ]
        self.assertEqual(counts["labeled"], 3)
        self.assertEqual(counts["pending"], 0)

    def test_non_pristine_monitor_cannot_bind_to_durable_run(self):
        monitor = CalibrationDriftMonitor(
            MonitorConfig(
                run_id="ghost",
                reference_size=1,
                current_size=1,
                min_current_size=1,
            )
        )
        monitor.track(
            observation("ghost"),
            support_labels=cyclic_law(2).labels,
        )
        with self.assertRaisesRegex(
            MonitorConfigurationConflictError, "must be empty"
        ):
            PosteriorMemoryHarness(
                SQLiteMemoryStore(self.path),
                monitor=monitor,
            )


if __name__ == "__main__":
    unittest.main()
