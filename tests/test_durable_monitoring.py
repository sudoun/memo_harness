import json
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from posterior_memory_harness import (
    CalibrationDriftMonitor,
    MemoryObservation,
    MonitorConfig,
    PosteriorMemoryHarness,
    SQLiteMemoryStore,
    cyclic_law,
    s3_law,
)
from posterior_memory_harness.store import (
    MonitorConfigurationConflictError,
    MonitorIdempotencyConflictError,
    MonitorOutcomeConflictError,
)


def config(**changes):
    values = {
        "reference_size": 1,
        "current_size": 1,
        "min_current_size": 1,
        "ece_bins": 5,
        "max_ece": 0.3,
        "max_tracked": 100,
    }
    values.update(changes)
    return MonitorConfig(**values)


def observation(identifier, posterior=(0.8, 0.2), *, observed_at=1.0):
    return MemoryObservation(
        observation_id=identifier,
        namespace="run",
        relation_type="binary",
        source=f"s-{identifier}",
        target=f"t-{identifier}",
        posterior=tuple(posterior),
        prior=tuple(1.0 / len(posterior) for _ in posterior),
        evidence_family=identifier,
        provenance={"monitor_family": "encoder-a"},
        observed_at=observed_at,
    )


def harness(path, monitor_config=None):
    return PosteriorMemoryHarness(
        SQLiteMemoryStore(path),
        monitor=CalibrationDriftMonitor(monitor_config or config()),
    ).register_relation("binary", cyclic_law(2))


class DurableMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite"

    def tearDown(self):
        self.temporary.cleanup()

    def test_restart_replays_an_identical_report(self):
        runtime = harness(self.path)
        now = time.time()
        for index in range(2):
            item = observation(
                f"obs-{index}",
                observed_at=now - 100 + index,
            )
            runtime.observe(item)
            runtime.record_outcome(
                item.observation_id,
                "0",
                labeled_at=now - 50 + index,
                recorded_at=now - 40 + index,
                outcome_event_id=f"event-{index}",
            )
        before = runtime.calibration_report("run")
        after = harness(self.path).calibration_report("run")
        self.assertEqual(before, after)
        self.assertEqual(
            after["relations"]["binary"]["overall"]["status"],
            "healthy",
        )

    def test_observation_and_prediction_roll_back_together(self):
        runtime = harness(self.path)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """
                CREATE TRIGGER fail_test_prediction
                BEFORE INSERT ON monitor_predictions
                BEGIN
                    SELECT RAISE(ABORT, 'test failpoint');
                END
                """
            )
        with self.assertRaisesRegex(ValueError, "transaction"):
            runtime.observe(observation("atomic"))
        with closing(sqlite3.connect(self.path)) as connection, connection:
            observations = connection.execute(
                "SELECT COUNT(*) FROM observations"
            ).fetchone()[0]
            predictions = connection.execute(
                "SELECT COUNT(*) FROM monitor_predictions"
            ).fetchone()[0]
        self.assertEqual((observations, predictions), (0, 0))
        self.assertEqual(runtime.monitor.tracked_count, 0)

    def test_correction_is_append_only_and_as_of_is_knowledge_safe(self):
        runtime = harness(self.path)
        now = time.time()
        for index in range(2):
            runtime.observe(
                observation(f"obs-{index}", observed_at=now - 100 + index)
            )
            runtime.record_outcome(
                f"obs-{index}",
                "0",
                labeled_at=now - 20 + index,
                recorded_at=now - 10 + index,
                outcome_event_id=f"initial-{index}",
            )
        runtime.correct_outcome(
            "obs-0",
            "1",
            reason="human verification",
            labeled_at=now - 20,
            recorded_at=now + 10,
            outcome_event_id="correction-0",
        )
        before = runtime.monitor.report(
            namespace="run",
            relation_type="binary",
            as_of=now + 5,
        )
        after = runtime.monitor.report(
            namespace="run",
            relation_type="binary",
            as_of=now + 20,
        )
        self.assertAlmostEqual(before["reference"]["accuracy"], 1.0)
        self.assertAlmostEqual(after["reference"]["accuracy"], 0.0)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            rows = connection.execute(
                """
                SELECT outcome_event_id, supersedes_event_id
                FROM monitor_outcomes
                WHERE observation_id = 'obs-0'
                ORDER BY outcome_seq
                """
            ).fetchall()
            self.assertEqual(
                rows,
                [("initial-0", None), ("correction-0", "initial-0")],
            )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    UPDATE monitor_outcomes SET true_index = 0
                    WHERE outcome_event_id = 'correction-0'
                    """
                )

    def test_locked_monitor_config_cannot_change_on_restart(self):
        harness(self.path, config(max_ece=0.1))
        with self.assertRaises(MonitorConfigurationConflictError):
            harness(self.path, config(max_ece=0.2))

    def test_registered_law_is_checked_against_durable_snapshot(self):
        runtime = PosteriorMemoryHarness(
            SQLiteMemoryStore(self.path),
            monitor=CalibrationDriftMonitor(config()),
        ).register_relation("permutation", s3_law())
        item = MemoryObservation(
            observation_id="s3",
            namespace="run",
            relation_type="permutation",
            source="a",
            target="b",
            posterior=(0.7, 0.06, 0.06, 0.06, 0.06, 0.06),
            prior=(1 / 6,) * 6,
            evidence_family="s3",
            observed_at=1.0,
        )
        runtime.observe(item)
        restarted = PosteriorMemoryHarness(
            SQLiteMemoryStore(self.path),
            monitor=CalibrationDriftMonitor(config()),
        )
        with self.assertRaises(MonitorConfigurationConflictError):
            restarted.register_relation(
                "permutation", s3_law().opposite("wrong-order")
            )

    def test_concurrent_first_outcome_has_one_durable_winner(self):
        creator = harness(self.path)
        creator.observe(observation("race"))
        left = harness(self.path)
        right = harness(self.path)

        def label(runtime, truth, event_id):
            try:
                return runtime.record_outcome(
                    "race",
                    truth,
                    outcome_event_id=event_id,
                )
            except MonitorOutcomeConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda values: label(*values),
                    (
                        (left, "0", "left"),
                        (right, "1", "right"),
                    ),
                )
            )
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count("conflict"), 1)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM monitor_outcomes"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_verify_checks_hashes_and_foreign_keys(self):
        runtime = harness(self.path)
        runtime.observe(observation("verified"))
        result = SQLiteMemoryStore(
            self.path, initialize=False
        ).verify_integrity()
        self.assertTrue(result["ok"])
        self.assertEqual(result["checks"]["monitor_predictions"], 1)

    def test_integer_timestamp_restarts_with_the_same_payload_hash(self):
        runtime = harness(self.path)
        runtime.observe(observation("integer-time", observed_at=1))
        runtime.observe(observation("negative-zero-time", observed_at=-0.0))
        self.assertTrue(
            SQLiteMemoryStore(
                self.path, initialize=False
            ).verify_integrity()["ok"]
        )
        restarted = harness(self.path)
        self.assertEqual(restarted.monitor.tracked_count, 2)

    def test_stale_worker_syncs_predictions_and_outcomes_before_health(self):
        stale = harness(self.path)
        writer = harness(self.path)
        now = time.time()
        for index in range(2):
            writer.observe(
                observation(f"external-{index}", observed_at=now - 10 + index)
            )
            writer.record_outcome(
                f"external-{index}",
                "0",
                outcome_event_id=f"external-event-{index}",
            )
        report = stale.calibration_report("run")
        overall = report["relations"]["binary"]["overall"]
        self.assertEqual(overall["counts"]["tracked"], 2)
        self.assertEqual(overall["counts"]["labeled"], 2)

    def test_first_registration_atomically_locks_relation_law(self):
        left = PosteriorMemoryHarness(
            SQLiteMemoryStore(self.path),
            monitor=CalibrationDriftMonitor(config()),
        )
        right = PosteriorMemoryHarness(
            SQLiteMemoryStore(self.path),
            monitor=CalibrationDriftMonitor(config()),
        )

        def register(runtime, law):
            try:
                runtime.register_relation("permutation", law)
                return True
            except MonitorConfigurationConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda values: register(*values),
                    (
                        (left, s3_law()),
                        (right, s3_law().opposite("wrong-order")),
                    ),
                )
            )
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count("conflict"), 1)
        with closing(sqlite3.connect(self.path)) as connection:
            count = connection.execute(
                """
                SELECT COUNT(*) FROM monitor_laws
                WHERE relation_type = 'permutation'
                """
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_relation_law_cannot_be_bypassed_with_another_run_or_no_monitor(self):
        owner = PosteriorMemoryHarness(
            SQLiteMemoryStore(self.path),
            monitor=CalibrationDriftMonitor(config(run_id="owner")),
        ).register_relation("permutation", s3_law())
        owner.observe(
            MemoryObservation(
                observation_id="owned",
                namespace="run",
                relation_type="permutation",
                source="a",
                target="b",
                posterior=(0.7, 0.06, 0.06, 0.06, 0.06, 0.06),
                prior=(1 / 6,) * 6,
                evidence_family="owned",
                observed_at=1,
            )
        )
        with self.assertRaises(MonitorConfigurationConflictError):
            PosteriorMemoryHarness(
                SQLiteMemoryStore(self.path),
                monitor=CalibrationDriftMonitor(config(run_id="bypass")),
            ).register_relation(
                "permutation", s3_law().opposite("wrong-run")
            )
        with self.assertRaises(MonitorConfigurationConflictError):
            PosteriorMemoryHarness(
                SQLiteMemoryStore(self.path)
            ).register_relation(
                "permutation", s3_law().opposite("unmonitored")
            )

    def test_durable_capacity_is_enforced_across_stale_workers(self):
        locked = config(max_tracked=2)
        workers = [harness(self.path, locked) for _ in range(3)]

        def append(index):
            try:
                workers[index].observe(observation(f"capacity-{index}"))
                return True
            except RuntimeError:
                return "full"

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(append, range(3)))
        self.assertEqual(results.count(True), 2)
        self.assertEqual(results.count("full"), 1)
        with closing(sqlite3.connect(self.path)) as connection:
            counts = (
                connection.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0],
                connection.execute(
                    "SELECT COUNT(*) FROM monitor_predictions"
                ).fetchone()[0],
            )
        self.assertEqual(counts, (2, 2))

    def test_restarted_as_of_report_preserves_pre_correction_truth(self):
        runtime = harness(self.path)
        now = time.time()
        for index in range(2):
            runtime.observe(
                observation(f"history-{index}", observed_at=now - 20 + index)
            )
            runtime.record_outcome(
                f"history-{index}",
                "0",
                labeled_at=now - 10 + index,
                recorded_at=now + index,
                outcome_event_id=f"history-initial-{index}",
            )
        runtime.correct_outcome(
            "history-0",
            "1",
            reason="verified correction",
            recorded_at=now + 10,
            outcome_event_id="history-correction",
        )
        restarted = harness(self.path)
        before = restarted.monitor.report(
            namespace="run",
            relation_type="binary",
            as_of=now + 5,
        )
        after = restarted.monitor.report(
            namespace="run",
            relation_type="binary",
            as_of=now + 20,
        )
        self.assertAlmostEqual(before["reference"]["accuracy"], 1.0)
        self.assertAlmostEqual(after["reference"]["accuracy"], 0.0)

    def test_verify_rejects_hash_tampering(self):
        runtime = harness(self.path)
        runtime.observe(observation("tampered"))
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "DROP TRIGGER monitor_predictions_no_update"
            )
            connection.execute(
                """
                UPDATE monitor_predictions
                SET payload_sha256 = ?
                WHERE observation_id = 'tampered'
                """,
                ("0" * 64,),
            )
            connection.execute(
                """
                CREATE TRIGGER monitor_predictions_no_update
                BEFORE UPDATE ON monitor_predictions
                BEGIN
                    SELECT RAISE(
                        ABORT, 'monitor_predictions is append-only'
                    );
                END
                """
            )
        result = SQLiteMemoryStore(
            self.path, initialize=False
        ).verify_integrity()
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("payload hash mismatch" in error for error in result["errors"])
        )

    def test_outcome_idempotency_and_correction_reason_are_type_safe(self):
        runtime = harness(self.path)
        runtime.observe(observation("idempotent"))
        self.assertTrue(
            runtime.record_outcome(
                "idempotent",
                "0",
                labeled_at=2,
                recorded_at=3,
                outcome_event_id="stable-event",
            )
        )
        self.assertFalse(
            runtime.record_outcome(
                "idempotent",
                "0",
                labeled_at=2,
                recorded_at=3,
                outcome_event_id="stable-event",
            )
        )
        with self.assertRaises(MonitorIdempotencyConflictError):
            runtime.record_outcome(
                "idempotent",
                "0",
                labeled_at=9,
                recorded_at=10,
                outcome_event_id="stable-event",
            )
        with self.assertRaisesRegex(ValueError, "must be a string"):
            runtime.correct_outcome(
                "idempotent",
                "1",
                reason=123,
                outcome_event_id="bad-reason",
            )
        restarted = harness(self.path)
        self.assertEqual(
            restarted.monitor.current_outcome(
                "idempotent"
            ).outcome_event_id,
            "stable-event",
        )


if __name__ == "__main__":
    unittest.main()
