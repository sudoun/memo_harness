import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from posterior_memory_harness import (
    CURRENT_SCHEMA_VERSION,
    MemoryObservation,
    MemoryQuery,
    SQLiteMemoryStore,
)
from posterior_memory_harness.sqlite_schema import (
    LEGACY_MIGRATION_NAME,
    LINEAGE_MIGRATION_NAME,
    SchemaMigrationError,
    UnsupportedSchemaVersionError,
    _create_monitor_schema,
    _ensure_history,
    _record_version,
)
from posterior_memory_harness.cli import schema_main


LEGACY_SCHEMA = """
CREATE TABLE observations (
    observation_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    posterior_json TEXT NOT NULL,
    prior_json TEXT NOT NULL,
    evidence_family TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    observed_at REAL NOT NULL,
    valid_from REAL,
    valid_until REAL,
    independent_evidence INTEGER NOT NULL,
    trust REAL NOT NULL,
    tags_json TEXT NOT NULL,
    revoked_at REAL
    {extra_column}
)
"""


def create_legacy_database(path: Path, *, extra_column: bool = False) -> None:
    extra = ", custom_metadata TEXT DEFAULT 'legacy'" if extra_column else ""
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(LEGACY_SCHEMA.format(extra_column=extra))
        connection.execute(
            """
            INSERT INTO observations (
                observation_id, namespace, relation_type, source, target,
                posterior_json, prior_json, evidence_family, provenance_json,
                observed_at, valid_from, valid_until, independent_evidence,
                trust, tags_json, revoked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-row",
                "agent",
                "phase",
                "a",
                "b",
                json.dumps([0.8, 0.2]),
                json.dumps([0.5, 0.5]),
                "legacy-event",
                json.dumps({"source": "v0.3"}),
                1.0,
                None,
                None,
                1,
                1.0,
                json.dumps(["project-x"]),
                None,
            ),
        )


def observation(identifier: str) -> MemoryObservation:
    return MemoryObservation(
        observation_id=identifier,
        namespace="agent",
        relation_type="phase",
        source="a",
        target="b",
        posterior=(0.8, 0.2),
        prior=(0.5, 0.5),
        evidence_family=identifier,
        observed_at=2.0,
    )


def query() -> MemoryQuery:
    return MemoryQuery("agent", "phase", ("a", "b"), "a", 10.0)


class SQLiteSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite"

    def tearDown(self):
        self.temporary.cleanup()

    def test_new_database_records_current_schema(self):
        status = SQLiteMemoryStore(self.path).schema_status()
        self.assertEqual(status.version, CURRENT_SCHEMA_VERSION)
        self.assertTrue(status.compatible)
        self.assertEqual(
            [item.version for item in status.migrations],
            [1, 2, 3],
        )
        self.assertIn("lineage_root_id", status.observation_columns)
        self.assertIn("observation_id", status.observation_columns)
        self.assertIn("run_id", status.monitor_run_columns)
        self.assertIn("law_fingerprint", status.relation_law_columns)
        self.assertIn("law_fingerprint", status.monitor_law_columns)
        self.assertIn(
            "law_fingerprint", status.monitor_prediction_columns
        )
        self.assertIn(
            "supersedes_event_id", status.monitor_outcome_columns
        )
        self.assertIn("monitor_outcomes_no_update", status.triggers)

    def test_legacy_v0_database_is_adopted_without_data_loss(self):
        create_legacy_database(self.path)
        store = SQLiteMemoryStore(self.path)
        status = store.schema_status()
        self.assertEqual(status.version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(status.migrations[0].name, LEGACY_MIGRATION_NAME)
        self.assertEqual(
            [item.version for item in status.migrations],
            [1, 2, 3],
        )
        restored = store.retrieve(query())
        self.assertEqual([item.observation_id for item in restored], ["legacy-row"])
        self.assertEqual(restored[0].provenance["source"], "v0.3")

    def test_extra_legacy_column_does_not_break_named_insert(self):
        create_legacy_database(self.path, extra_column=True)
        store = SQLiteMemoryStore(self.path)
        store.append(observation("new-row"))
        self.assertEqual(
            [item.observation_id for item in store.retrieve(query())],
            ["new-row", "legacy-row"],
        )
        with closing(sqlite3.connect(self.path)) as connection, connection:
            value = connection.execute(
                """
                SELECT custom_metadata FROM observations
                WHERE observation_id = 'new-row'
                """
            ).fetchone()[0]
        self.assertEqual(value, "legacy")

    def test_version_one_database_migrates_to_current_without_data_loss(self):
        create_legacy_database(self.path)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE memory_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO memory_schema_migrations
                VALUES (1, 'baseline_observations_v1', 'locked')
                """
            )
            connection.execute("PRAGMA user_version = 1")
        store = SQLiteMemoryStore(self.path)
        status = store.schema_status()
        self.assertEqual(status.version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(
            [item.version for item in status.migrations],
            [1, 2, 3],
        )
        self.assertEqual(
            [item.observation_id for item in store.retrieve(query())],
            ["legacy-row"],
        )

    def test_version_two_database_adds_lineage_columns_without_data_loss(self):
        create_legacy_database(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            _ensure_history(connection)
            _record_version(connection, 1, "baseline_observations_v1")
            _create_monitor_schema(connection)
            _record_version(connection, 2, "durable_monitor_journal_v2")
            connection.execute("PRAGMA user_version = 2")
            connection.commit()

        store = SQLiteMemoryStore(self.path)
        status = store.schema_status()
        self.assertEqual(status.version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(status.migrations[-1].name, LINEAGE_MIGRATION_NAME)
        restored = store.retrieve(query())[0]
        self.assertEqual(restored.observation_id, "legacy-row")
        self.assertIsNone(restored.source_event_id)
        self.assertIsNone(restored.lineage_root_id)
        self.assertEqual(restored.derived_from, ())

    def test_future_schema_is_rejected_without_downgrade(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA user_version = 999")
        with self.assertRaisesRegex(UnsupportedSchemaVersionError, "newer"):
            SQLiteMemoryStore(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, 999)

    def test_malformed_legacy_schema_fails_without_destructive_rebuild(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TABLE observations(observation_id TEXT PRIMARY KEY)"
            )
        with self.assertRaisesRegex(SchemaMigrationError, "missing required"):
            SQLiteMemoryStore(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = [
                row[1]
                for row in connection.execute("PRAGMA table_info(observations)")
            ]
        self.assertEqual(version, 0)
        self.assertEqual(columns, ["observation_id"])

    def test_reopening_current_database_is_idempotent(self):
        first = SQLiteMemoryStore(self.path).schema_status()
        second = SQLiteMemoryStore(self.path).schema_status()
        self.assertEqual(first.migrations, second.migrations)
        self.assertEqual(len(second.migrations), 3)

    def test_concurrent_first_open_applies_migration_once(self):
        def open_store(_):
            return SQLiteMemoryStore(self.path).schema_status().version

        with ThreadPoolExecutor(max_workers=4) as executor:
            versions = list(executor.map(open_store, range(8)))
        self.assertEqual(versions, [CURRENT_SCHEMA_VERSION] * 8)
        status = SQLiteMemoryStore(self.path).schema_status()
        self.assertEqual(len(status.migrations), 3)

    def test_schema_cli_emits_machine_readable_status(self):
        SQLiteMemoryStore(self.path)
        output = StringIO()
        with redirect_stdout(output):
            schema_main(["--db", str(self.path)])
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["version"], CURRENT_SCHEMA_VERSION)
        self.assertTrue(payload["compatible"])
        self.assertEqual(payload["migrations"][0]["version"], 1)
        self.assertEqual(payload["migrations"][1]["version"], 2)
        self.assertEqual(payload["migrations"][2]["version"], 3)


if __name__ == "__main__":
    unittest.main()
