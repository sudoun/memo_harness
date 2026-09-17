"""Versioned, auditable SQLite schema management."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sqlite3


CURRENT_SCHEMA_VERSION = 3
BASELINE_MIGRATION_NAME = "baseline_observations_v1"
LEGACY_MIGRATION_NAME = "adopt_legacy_observations_v0"
MONITOR_MIGRATION_NAME = "durable_monitor_journal_v2"
LINEAGE_MIGRATION_NAME = "evidence_lineage_v3"

OBSERVATION_V1_COLUMNS = (
    "observation_id",
    "namespace",
    "relation_type",
    "source",
    "target",
    "posterior_json",
    "prior_json",
    "evidence_family",
    "provenance_json",
    "observed_at",
    "valid_from",
    "valid_until",
    "independent_evidence",
    "trust",
    "tags_json",
    "revoked_at",
)

OBSERVATION_COLUMNS = OBSERVATION_V1_COLUMNS + (
    "source_event_id",
    "lineage_root_id",
    "derived_from_json",
    "encoder_revision",
    "calibration_revision",
)

MONITOR_RUN_COLUMNS = (
    "run_id",
    "config_json",
    "config_sha256",
    "created_at",
)

RELATION_LAW_COLUMNS = (
    "relation_type",
    "support_labels_json",
    "law_fingerprint",
    "registered_at",
)

MONITOR_LAW_COLUMNS = (
    "run_id",
    "relation_type",
    "support_labels_json",
    "law_fingerprint",
    "registered_at",
)

MONITOR_PREDICTION_COLUMNS = (
    "prediction_seq",
    "observation_id",
    "run_id",
    "namespace",
    "relation_type",
    "source_family",
    "posterior_json",
    "support_labels_json",
    "support_size",
    "law_fingerprint",
    "observed_at",
    "tracked_at",
    "payload_sha256",
)

MONITOR_OUTCOME_COLUMNS = (
    "outcome_seq",
    "outcome_event_id",
    "observation_id",
    "relation_type",
    "true_index",
    "true_label",
    "law_fingerprint",
    "labeled_at",
    "recorded_at",
    "supersedes_event_id",
    "correction_reason",
    "payload_sha256",
)

REQUIRED_V2_TRIGGERS = (
    "memory_schema_migrations_no_update",
    "memory_schema_migrations_no_delete",
    "relation_laws_no_update",
    "relation_laws_no_delete",
    "monitor_laws_validate_global",
    "monitor_laws_no_update",
    "monitor_laws_no_delete",
    "monitor_predictions_validate_law",
    "monitor_predictions_no_update",
    "monitor_predictions_no_delete",
    "monitor_outcomes_validate",
    "monitor_outcomes_no_update",
    "monitor_outcomes_no_delete",
    "monitor_runs_no_update",
    "monitor_runs_no_delete",
)

REQUIRED_V2_INDEXES = (
    "idx_monitor_prediction_scope",
    "idx_monitor_outcome_observation",
    "idx_monitor_outcome_root",
    "idx_monitor_outcome_supersedes",
)


class SchemaMigrationError(RuntimeError):
    pass


class UnsupportedSchemaVersionError(SchemaMigrationError):
    pass


@dataclass(frozen=True)
class SchemaMigrationRecord:
    version: int
    name: str
    applied_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "name": self.name,
            "applied_at": self.applied_at,
        }


@dataclass(frozen=True)
class SQLiteSchemaStatus:
    version: int
    current_version: int
    observation_columns: tuple[str, ...]
    monitor_run_columns: tuple[str, ...]
    relation_law_columns: tuple[str, ...]
    monitor_law_columns: tuple[str, ...]
    monitor_prediction_columns: tuple[str, ...]
    monitor_outcome_columns: tuple[str, ...]
    migrations: tuple[SchemaMigrationRecord, ...]
    triggers: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return self.version == self.current_version

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "current_version": self.current_version,
            "compatible": self.compatible,
            "observation_columns": list(self.observation_columns),
            "monitor_run_columns": list(self.monitor_run_columns),
            "relation_law_columns": list(self.relation_law_columns),
            "monitor_law_columns": list(self.monitor_law_columns),
            "monitor_prediction_columns": list(
                self.monitor_prediction_columns
            ),
            "monitor_outcome_columns": list(self.monitor_outcome_columns),
            "migrations": [record.to_dict() for record in self.migrations],
            "triggers": list(self.triggers),
        }


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table,),
    ).fetchone()
    return row is not None


def _object_names(
    connection: sqlite3.Connection, object_type: str
) -> tuple[str, ...]:
    return tuple(
        str(row["name"])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = ?
            ORDER BY name
            """,
            (object_type,),
        )
    )


def _normalized_sql(value: str | None) -> str:
    if value is None:
        return ""
    # SQLite stores the optional creation modifier verbatim even though it has
    # no effect on the resulting object contract.
    return "".join(value.lower().split()).replace("ifnotexists", "")


def _object_sql(
    connection: sqlite3.Connection,
    object_type: str,
    name: str,
) -> str | None:
    row = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = ? AND name = ?
        """,
        (object_type, name),
    ).fetchone()
    return None if row is None else row["sql"]


def _table_columns(
    connection: sqlite3.Connection, table: str
) -> tuple[sqlite3.Row, ...]:
    return tuple(connection.execute(f"PRAGMA table_info({table})"))


def _column_names(
    connection: sqlite3.Connection, table: str
) -> tuple[str, ...]:
    if not _table_exists(connection, table):
        return ()
    return tuple(str(row["name"]) for row in _table_columns(connection, table))


def _validate_table(
    connection: sqlite3.Connection,
    table: str,
    required_columns: tuple[str, ...],
    primary_keys: set[str],
) -> tuple[str, ...]:
    rows = _table_columns(connection, table)
    if not rows:
        raise SchemaMigrationError(f"{table} table is missing")
    columns = tuple(str(row["name"]) for row in rows)
    missing = sorted(set(required_columns) - set(columns))
    if missing:
        raise SchemaMigrationError(
            f"{table} table is missing required columns: "
            + ", ".join(missing)
        )
    actual_primary_keys = {
        str(row["name"]) for row in rows if int(row["pk"]) > 0
    }
    if actual_primary_keys != primary_keys:
        expected = ", ".join(sorted(primary_keys))
        raise SchemaMigrationError(
            f"{table} primary key must be {expected}"
        )
    return columns


def _validate_observations(
    connection: sqlite3.Connection,
    required_columns: tuple[str, ...] = OBSERVATION_COLUMNS,
) -> tuple[str, ...]:
    return _validate_table(
        connection,
        "observations",
        required_columns,
        {"observation_id"},
    )


def _add_lineage_columns(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE observations ADD COLUMN source_event_id TEXT"
    )
    connection.execute(
        "ALTER TABLE observations ADD COLUMN lineage_root_id TEXT"
    )
    connection.execute(
        """
        ALTER TABLE observations
        ADD COLUMN derived_from_json TEXT NOT NULL DEFAULT '[]'
            CHECK(json_valid(derived_from_json)
                  AND json_type(derived_from_json) = 'array')
        """
    )
    connection.execute(
        "ALTER TABLE observations ADD COLUMN encoder_revision TEXT"
    )
    connection.execute(
        "ALTER TABLE observations ADD COLUMN calibration_revision TEXT"
    )


def _validate_v2(connection: sqlite3.Connection) -> None:
    _validate_table(
        connection,
        "monitor_runs",
        MONITOR_RUN_COLUMNS,
        {"run_id"},
    )
    _validate_table(
        connection,
        "relation_laws",
        RELATION_LAW_COLUMNS,
        {"relation_type"},
    )
    _validate_table(
        connection,
        "monitor_laws",
        MONITOR_LAW_COLUMNS,
        {"run_id", "relation_type"},
    )
    _validate_table(
        connection,
        "monitor_predictions",
        MONITOR_PREDICTION_COLUMNS,
        {"prediction_seq"},
    )
    _validate_table(
        connection,
        "monitor_outcomes",
        MONITOR_OUTCOME_COLUMNS,
        {"outcome_seq"},
    )
    trigger_names = set(_object_names(connection, "trigger"))
    missing_triggers = sorted(set(REQUIRED_V2_TRIGGERS) - trigger_names)
    if missing_triggers:
        raise SchemaMigrationError(
            "schema v2 is missing required triggers: "
            + ", ".join(missing_triggers)
        )
    attached_triggers = {
        str(row["name"])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'trigger'
              AND tbl_name IN (
                  'memory_schema_migrations',
                  'monitor_runs',
                  'relation_laws',
                  'monitor_laws',
                  'monitor_predictions',
                  'monitor_outcomes'
              )
            """
        )
    }
    unexpected_triggers = sorted(
        attached_triggers - set(REQUIRED_V2_TRIGGERS)
    )
    if unexpected_triggers:
        raise SchemaMigrationError(
            "schema v2 has unexpected triggers on locked tables: "
            + ", ".join(unexpected_triggers)
        )
    index_names = set(_object_names(connection, "index"))
    missing_indexes = sorted(set(REQUIRED_V2_INDEXES) - index_names)
    if missing_indexes:
        raise SchemaMigrationError(
            "schema v2 is missing required indexes: "
            + ", ".join(missing_indexes)
        )
    for (object_type, name), expected_sql in _reference_v2_sql().items():
        actual_sql = _object_sql(connection, object_type, name)
        if actual_sql is None:
            raise SchemaMigrationError(
                f"schema v2 is missing required {object_type} {name}"
            )
        if _normalized_sql(actual_sql) != expected_sql:
            raise SchemaMigrationError(
                f"schema v2 {object_type} {name} does not match the "
                "locked v2 contract"
            )


def _create_observations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
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
        )
        """
    )


def _ensure_observation_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_observation_scope
        ON observations(namespace, relation_type, observed_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_observation_nodes
        ON observations(source, target)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_observation_family
        ON observations(evidence_family)
        """
    )


def _create_monitor_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE monitor_runs (
            run_id TEXT PRIMARY KEY,
            config_json TEXT NOT NULL
                CHECK(json_valid(config_json)
                      AND json_type(config_json) = 'object'),
            config_sha256 TEXT NOT NULL CHECK(length(config_sha256) = 64),
            created_at REAL NOT NULL
                CHECK(created_at = created_at
                      AND created_at BETWEEN -1e308 AND 1e308)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE relation_laws (
            relation_type TEXT PRIMARY KEY,
            support_labels_json TEXT NOT NULL
                CHECK(json_valid(support_labels_json)
                      AND json_type(support_labels_json) = 'array'),
            law_fingerprint TEXT NOT NULL
                CHECK(length(law_fingerprint) = 64),
            registered_at REAL NOT NULL
                CHECK(registered_at = registered_at
                      AND registered_at BETWEEN -1e308 AND 1e308)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE monitor_laws (
            run_id TEXT NOT NULL
                REFERENCES monitor_runs(run_id) ON DELETE RESTRICT,
            relation_type TEXT NOT NULL,
            support_labels_json TEXT NOT NULL
                CHECK(json_valid(support_labels_json)
                      AND json_type(support_labels_json) = 'array'),
            law_fingerprint TEXT NOT NULL
                CHECK(length(law_fingerprint) = 64),
            registered_at REAL NOT NULL
                CHECK(registered_at = registered_at
                      AND registered_at BETWEEN -1e308 AND 1e308),
            PRIMARY KEY(run_id, relation_type),
            FOREIGN KEY(relation_type)
                REFERENCES relation_laws(relation_type) ON DELETE RESTRICT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE monitor_predictions (
            prediction_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_id TEXT NOT NULL UNIQUE
                REFERENCES observations(observation_id) ON DELETE RESTRICT,
            run_id TEXT NOT NULL
                REFERENCES monitor_runs(run_id) ON DELETE RESTRICT,
            namespace TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            source_family TEXT NOT NULL
                CHECK(length(source_family) BETWEEN 1 AND 128),
            posterior_json TEXT NOT NULL
                CHECK(json_valid(posterior_json)
                      AND json_type(posterior_json) = 'array'),
            support_labels_json TEXT NOT NULL
                CHECK(json_valid(support_labels_json)
                      AND json_type(support_labels_json) = 'array'),
            support_size INTEGER NOT NULL
                CHECK(support_size >= 1
                      AND support_size = json_array_length(posterior_json)
                      AND support_size =
                          json_array_length(support_labels_json)),
            law_fingerprint TEXT NOT NULL
                CHECK(length(law_fingerprint) = 64),
            observed_at REAL NOT NULL
                CHECK(observed_at = observed_at
                      AND observed_at BETWEEN -1e308 AND 1e308),
            tracked_at REAL NOT NULL
                CHECK(tracked_at = tracked_at
                      AND tracked_at BETWEEN -1e308 AND 1e308),
            payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64),
            FOREIGN KEY(run_id, relation_type)
                REFERENCES monitor_laws(run_id, relation_type)
                ON DELETE RESTRICT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE monitor_outcomes (
            outcome_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            outcome_event_id TEXT NOT NULL UNIQUE,
            observation_id TEXT NOT NULL
                REFERENCES monitor_predictions(observation_id)
                ON DELETE RESTRICT,
            relation_type TEXT NOT NULL,
            true_index INTEGER NOT NULL CHECK(true_index >= 0),
            true_label TEXT NOT NULL,
            law_fingerprint TEXT NOT NULL
                CHECK(length(law_fingerprint) = 64),
            labeled_at REAL NOT NULL
                CHECK(labeled_at = labeled_at
                      AND labeled_at BETWEEN -1e308 AND 1e308),
            recorded_at REAL NOT NULL
                CHECK(recorded_at = recorded_at
                      AND recorded_at BETWEEN -1e308 AND 1e308),
            supersedes_event_id TEXT UNIQUE
                REFERENCES monitor_outcomes(outcome_event_id)
                ON DELETE RESTRICT,
            correction_reason TEXT,
            payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_monitor_prediction_scope
        ON monitor_predictions(
            run_id, namespace, relation_type, source_family,
            observed_at, observation_id
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_monitor_outcome_observation
        ON monitor_outcomes(observation_id, recorded_at, outcome_seq)
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_monitor_outcome_root
        ON monitor_outcomes(observation_id)
        WHERE supersedes_event_id IS NULL
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_monitor_outcome_supersedes
        ON monitor_outcomes(supersedes_event_id)
        WHERE supersedes_event_id IS NOT NULL
        """
    )
    connection.execute(
        """
        CREATE TRIGGER monitor_laws_validate_global
        BEFORE INSERT ON monitor_laws
        BEGIN
            SELECT CASE WHEN NOT EXISTS (
                SELECT 1 FROM relation_laws AS global_law
                WHERE global_law.relation_type = NEW.relation_type
                  AND global_law.support_labels_json =
                      NEW.support_labels_json
                  AND global_law.law_fingerprint =
                      NEW.law_fingerprint
            ) THEN RAISE(
                ABORT, 'monitor law differs from durable relation law'
            ) END;
            SELECT CASE WHEN EXISTS (
                SELECT 1 FROM monitor_laws AS existing
                WHERE existing.relation_type = NEW.relation_type
                  AND (
                      existing.support_labels_json <>
                          NEW.support_labels_json
                      OR existing.law_fingerprint <>
                          NEW.law_fingerprint
                  )
            ) THEN RAISE(
                ABORT, 'relation law differs across monitor runs'
            ) END;
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER monitor_predictions_validate_law
        BEFORE INSERT ON monitor_predictions
        BEGIN
            SELECT CASE WHEN NOT EXISTS (
                SELECT 1 FROM monitor_laws AS law
                WHERE law.run_id = NEW.run_id
                  AND law.relation_type = NEW.relation_type
                  AND law.support_labels_json = NEW.support_labels_json
                  AND law.law_fingerprint = NEW.law_fingerprint
            ) THEN RAISE(ABORT, 'monitor relation law mismatch') END;
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER monitor_outcomes_validate
        BEFORE INSERT ON monitor_outcomes
        BEGIN
            SELECT CASE WHEN NOT EXISTS (
                SELECT 1 FROM monitor_predictions AS p
                WHERE p.observation_id = NEW.observation_id
                  AND p.relation_type = NEW.relation_type
                  AND p.law_fingerprint = NEW.law_fingerprint
                  AND NEW.true_index < p.support_size
                  AND json_extract(
                        p.support_labels_json,
                        '$[' || NEW.true_index || ']'
                      ) = NEW.true_label
                  AND NEW.labeled_at >= p.observed_at
            ) THEN RAISE(ABORT, 'invalid monitor outcome') END;
            SELECT CASE WHEN NEW.supersedes_event_id IS NULL
                AND NEW.correction_reason IS NOT NULL
                THEN RAISE(ABORT, 'initial outcome cannot be a correction') END;
            SELECT CASE WHEN NEW.supersedes_event_id IS NOT NULL
                AND (
                    NEW.correction_reason IS NULL
                    OR length(trim(NEW.correction_reason)) = 0
                )
                THEN RAISE(ABORT, 'correction reason is required') END;
            SELECT CASE WHEN NEW.supersedes_event_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM monitor_outcomes AS previous
                    WHERE previous.outcome_event_id =
                          NEW.supersedes_event_id
                      AND previous.observation_id = NEW.observation_id
                      AND previous.recorded_at <= NEW.recorded_at
                      AND previous.labeled_at <= NEW.labeled_at
                )
                THEN RAISE(ABORT, 'invalid correction predecessor') END;
        END
        """
    )
    for table in (
        "relation_laws",
        "monitor_runs",
        "monitor_laws",
        "monitor_predictions",
        "monitor_outcomes",
    ):
        connection.execute(
            f"""
            CREATE TRIGGER {table}_no_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{table} is append-only');
            END
            """
        )
        connection.execute(
            f"""
            CREATE TRIGGER {table}_no_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{table} is append-only');
            END
            """
        )


@lru_cache(maxsize=1)
def _reference_v2_sql() -> dict[tuple[str, str], str]:
    """Build the canonical v2 contract with the active SQLite engine.

    Comparing complete normalized DDL catches a same-name no-op trigger,
    changed CHECK/FK/UNIQUE semantics, or an index whose columns were replaced.
    The observation table is intentionally excluded because compatible legacy
    deployments may retain additive application columns.
    """
    reference = sqlite3.connect(":memory:")
    reference.row_factory = sqlite3.Row
    try:
        reference.execute("PRAGMA foreign_keys=ON")
        _create_observations(reference)
        _create_monitor_schema(reference)
        _ensure_history(reference)
        contracts: dict[tuple[str, str], str] = {}
        for table in (
            "memory_schema_migrations",
            "monitor_runs",
            "relation_laws",
            "monitor_laws",
            "monitor_predictions",
            "monitor_outcomes",
        ):
            contracts[("table", table)] = _normalized_sql(
                _object_sql(reference, "table", table)
            )
        for trigger in REQUIRED_V2_TRIGGERS:
            contracts[("trigger", trigger)] = _normalized_sql(
                _object_sql(reference, "trigger", trigger)
            )
        for index in REQUIRED_V2_INDEXES:
            contracts[("index", index)] = _normalized_sql(
                _object_sql(reference, "index", index)
            )
        return contracts
    finally:
        reference.close()


def _ensure_history(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_schema_migrations_no_update
        BEFORE UPDATE ON memory_schema_migrations
        BEGIN
            SELECT RAISE(
                ABORT, 'memory_schema_migrations is append-only'
            );
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_schema_migrations_no_delete
        BEFORE DELETE ON memory_schema_migrations
        BEGIN
            SELECT RAISE(
                ABORT, 'memory_schema_migrations is append-only'
            );
        END
        """
    )


def _record_version(
    connection: sqlite3.Connection, version: int, name: str
) -> None:
    existing = connection.execute(
        """
        SELECT name FROM memory_schema_migrations WHERE version = ?
        """,
        (version,),
    ).fetchone()
    if existing is not None:
        if str(existing["name"]) != name:
            raise SchemaMigrationError(
                f"migration version {version} is recorded as "
                f"{existing['name']!r}, expected {name!r}"
            )
        return
    connection.execute(
        """
        INSERT INTO memory_schema_migrations(version, name, applied_at)
        VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        """,
        (version, name),
    )


def migrate_schema(connection: sqlite3.Connection) -> None:
    """Atomically create or migrate the durable harness schema."""
    try:
        connection.execute("BEGIN IMMEDIATE")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CURRENT_SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"database schema {version} is newer than supported "
                f"version {CURRENT_SCHEMA_VERSION}"
            )

        if version == 0:
            if _table_exists(connection, "observations"):
                _validate_observations(
                    connection,
                    OBSERVATION_V1_COLUMNS,
                )
                migration_name = LEGACY_MIGRATION_NAME
            else:
                _create_observations(connection)
                migration_name = BASELINE_MIGRATION_NAME
            _ensure_observation_indexes(connection)
            _ensure_history(connection)
            _record_version(connection, 1, migration_name)
            connection.execute("PRAGMA user_version = 1")
            version = 1

        if version == 1:
            _validate_observations(
                connection,
                OBSERVATION_V1_COLUMNS,
            )
            _ensure_observation_indexes(connection)
            _ensure_history(connection)
            existing_v1 = connection.execute(
                """
                SELECT name FROM memory_schema_migrations WHERE version = 1
                """
            ).fetchone()
            if existing_v1 is None:
                _record_version(connection, 1, BASELINE_MIGRATION_NAME)
            if any(
                _table_exists(connection, table)
                for table in (
                    "relation_laws",
                    "monitor_runs",
                    "monitor_laws",
                    "monitor_predictions",
                    "monitor_outcomes",
                )
            ):
                raise SchemaMigrationError(
                    "partial monitor schema exists before migration v2"
                )
            _create_monitor_schema(connection)
            _record_version(connection, 2, MONITOR_MIGRATION_NAME)
            connection.execute("PRAGMA user_version = 2")
            version = 2

        if version == 2:
            _validate_observations(
                connection,
                OBSERVATION_V1_COLUMNS,
            )
            if not _table_exists(connection, "memory_schema_migrations"):
                raise SchemaMigrationError(
                    "schema v2 migration history table is missing"
                )
            _validate_v2(connection)
            _add_lineage_columns(connection)
            _record_version(connection, 3, LINEAGE_MIGRATION_NAME)
            connection.execute("PRAGMA user_version = 3")
            version = 3

        if version == 3:
            _validate_observations(connection)
            if not _table_exists(connection, "memory_schema_migrations"):
                raise SchemaMigrationError(
                    "schema v3 migration history table is missing"
                )
            _validate_v2(connection)
        else:
            raise UnsupportedSchemaVersionError(
                f"no migration path from schema version {version}"
            )

        final_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if final_version != CURRENT_SCHEMA_VERSION:
            raise SchemaMigrationError(
                f"migration ended at version {final_version}, expected "
                f"{CURRENT_SCHEMA_VERSION}"
            )
        validate_current_schema(connection)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def validate_current_schema(connection: sqlite3.Connection) -> None:
    """Fail closed if a purported current database is structurally incomplete."""
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != CURRENT_SCHEMA_VERSION:
        raise SchemaMigrationError(
            f"schema version {version} is not current "
            f"({CURRENT_SCHEMA_VERSION})"
        )
    _validate_observations(connection)
    _validate_v2(connection)
    if not _table_exists(connection, "memory_schema_migrations"):
        raise SchemaMigrationError("migration history table is missing")
    rows = {
        int(row["version"]): str(row["name"])
        for row in connection.execute(
            """
            SELECT version, name FROM memory_schema_migrations
            """
        )
    }
    if set(rows) != {1, 2, 3}:
        raise SchemaMigrationError(
            "migration history must contain exactly versions 1, 2, and 3"
        )
    if rows.get(1) not in {
        BASELINE_MIGRATION_NAME,
        LEGACY_MIGRATION_NAME,
    }:
        raise SchemaMigrationError("migration v1 history is missing or invalid")
    if rows.get(2) != MONITOR_MIGRATION_NAME:
        raise SchemaMigrationError("migration v2 history is missing or invalid")
    if rows.get(3) != LINEAGE_MIGRATION_NAME:
        raise SchemaMigrationError("migration v3 history is missing or invalid")


def read_schema_status(connection: sqlite3.Connection) -> SQLiteSchemaStatus:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    migrations = ()
    if _table_exists(connection, "memory_schema_migrations"):
        migrations = tuple(
            SchemaMigrationRecord(
                version=int(row["version"]),
                name=str(row["name"]),
                applied_at=str(row["applied_at"]),
            )
            for row in connection.execute(
                """
                SELECT version, name, applied_at
                FROM memory_schema_migrations
                ORDER BY version
                """
            )
        )
    return SQLiteSchemaStatus(
        version=version,
        current_version=CURRENT_SCHEMA_VERSION,
        observation_columns=_column_names(connection, "observations"),
        monitor_run_columns=_column_names(connection, "monitor_runs"),
        relation_law_columns=_column_names(connection, "relation_laws"),
        monitor_law_columns=_column_names(connection, "monitor_laws"),
        monitor_prediction_columns=_column_names(
            connection, "monitor_predictions"
        ),
        monitor_outcome_columns=_column_names(connection, "monitor_outcomes"),
        migrations=migrations,
        triggers=_object_names(connection, "trigger"),
    )
