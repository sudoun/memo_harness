"""Append-only memory stores."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Mapping, Protocol

from .models import (
    MemoryNeighborhood,
    MemoryNeighborhoodQuery,
    MemoryObservation,
    MemoryQuery,
)
from .monitoring import (
    CalibrationDriftMonitor,
    MonitorConfig,
    MonitorIdempotencyConflictError,
    MonitorOutcomeRecord,
    MonitorOutcomeConflictError,
    MonitorPredictionRecord,
    validate_relation_law_snapshot,
)
from .sqlite_schema import (
    SQLiteSchemaStatus,
    migrate_schema,
    read_schema_status,
    validate_current_schema,
)
from .retrieval import (
    discover_structured_neighborhood,
    select_budgeted_observations,
)


class DuplicateObservationError(ValueError):
    pass


class ObservationNotFoundError(KeyError):
    pass


class MonitorConfigurationConflictError(ValueError):
    pass


class MonitorObservationNotTrackedError(KeyError):
    pass


class MemoryStore(Protocol):
    def append(self, observation: MemoryObservation) -> None: ...

    def revoke(self, observation_id: str, revoked_at: float) -> None: ...

    def retrieve(self, query: MemoryQuery) -> list[MemoryObservation]: ...

    def retrieve_neighborhood(
        self, query: MemoryNeighborhoodQuery
    ) -> MemoryNeighborhood: ...


def _validate_lineage_parent(
    observation: MemoryObservation,
    parent: MemoryObservation,
) -> None:
    if (
        observation.namespace != parent.namespace
        or observation.relation_type != parent.relation_type
    ):
        raise ValueError(
            "derived_from must remain within one namespace and relation type"
        )
    if observation.observed_at < parent.observed_at:
        raise ValueError("derived observation cannot precede its parent")
    if observation.independent_evidence:
        if observation.lineage_key != parent.lineage_key:
            raise ValueError(
                "an independent derived observation must retain its "
                "parent lineage root"
            )
        if (
            observation.source != parent.source
            or observation.target != parent.target
        ):
            raise ValueError(
                "an independent derived observation must retain its "
                "parent relation endpoints"
            )


def _is_active(observation: MemoryObservation, query: MemoryQuery) -> bool:
    if observation.namespace != query.namespace:
        return False
    if observation.relation_type != query.relation_type:
        return False
    if observation.source not in query.nodes or observation.target not in query.nodes:
        return False
    if observation.observed_at > query.as_of:
        return False
    if observation.revoked_at is not None and observation.revoked_at <= query.as_of:
        return False
    if observation.valid_from is not None and observation.valid_from > query.as_of:
        return False
    if observation.valid_until is not None and observation.valid_until < query.as_of:
        return False
    if not set(query.tags).issubset(observation.tags):
        return False
    return True


def _is_active_in_neighborhood(
    observation: MemoryObservation, query: MemoryNeighborhoodQuery
) -> bool:
    if observation.namespace != query.namespace:
        return False
    if observation.relation_type != query.relation_type:
        return False
    if not observation.independent_evidence:
        return False
    if observation.observed_at > query.as_of:
        return False
    if observation.revoked_at is not None and observation.revoked_at <= query.as_of:
        return False
    if observation.valid_from is not None and observation.valid_from > query.as_of:
        return False
    if observation.valid_until is not None and observation.valid_until < query.as_of:
        return False
    if not set(query.tags).issubset(observation.tags):
        return False
    return True


def _latest_evidence_families(
    observations: list[MemoryObservation],
) -> list[MemoryObservation]:
    latest: dict[str, MemoryObservation] = {}
    for observation in observations:
        previous = latest.get(observation.lineage_key)
        key = (observation.observed_at, observation.observation_id)
        if previous is None or key > (previous.observed_at, previous.observation_id):
            latest[observation.lineage_key] = observation
    return sorted(
        latest.values(),
        key=lambda item: (-item.observed_at, item.observation_id),
    )


def _bounded_neighborhood(
    observations: list[MemoryObservation], query: MemoryNeighborhoodQuery
) -> MemoryNeighborhood:
    active = [
        observation
        for observation in observations
        if _is_active_in_neighborhood(observation, query)
    ]
    if query.retrieval_policy != "recent":
        return discover_structured_neighborhood(active, query)
    candidates = _latest_evidence_families(
        active
    )
    nodes = list(query.seeds)
    node_set = set(nodes)
    frontier = set(query.seeds)
    selected: dict[str, MemoryObservation] = {}
    node_limit_hit = False
    observation_limit_hit = False
    hops_explored = 0

    for hop in range(1, query.max_hops + 1):
        incident = [
            observation
            for observation in candidates
            if observation.observation_id not in selected
            and (
                observation.source in frontier
                or observation.target in frontier
            )
        ]
        if not incident:
            break
        next_frontier: set[str] = set()
        for observation in incident:
            if len(selected) >= query.max_observations:
                observation_limit_hit = True
                break
            new_nodes = [
                node
                for node in (observation.source, observation.target)
                if node not in node_set
            ]
            if len(node_set) + len(new_nodes) > query.max_nodes:
                node_limit_hit = True
                continue
            selected[observation.observation_id] = observation
            for node in new_nodes:
                node_set.add(node)
                nodes.append(node)
                next_frontier.add(node)
        hops_explored = hop
        if observation_limit_hit or not next_frontier:
            break
        frontier = next_frontier

    return MemoryNeighborhood(
        nodes=tuple(nodes),
        observations=tuple(
            sorted(
                selected.values(),
                key=lambda item: (item.observed_at, item.observation_id),
            )
        ),
        hops_explored=hops_explored,
        node_limit_hit=node_limit_hit,
        observation_limit_hit=observation_limit_hit,
    )


class InMemoryMemoryStore:
    def __init__(self):
        self._observations: dict[str, MemoryObservation] = {}
        self._lock = threading.RLock()

    def append(self, observation: MemoryObservation) -> None:
        with self._lock:
            if observation.observation_id in self._observations:
                raise DuplicateObservationError(observation.observation_id)
            for parent_id in observation.derived_from:
                parent = self._observations.get(parent_id)
                if parent is None:
                    raise ValueError(
                        f"derived_from parent {parent_id!r} does not exist"
                    )
                _validate_lineage_parent(observation, parent)
            self._observations[observation.observation_id] = observation

    def revoke(self, observation_id: str, revoked_at: float) -> None:
        with self._lock:
            try:
                observation = self._observations[observation_id]
            except KeyError as error:
                raise ObservationNotFoundError(observation_id) from error
            if observation.revoked_at is not None:
                raise ObservationNotFoundError(observation_id)
            if float(revoked_at) < observation.observed_at:
                raise ValueError("revoked_at must not precede observed_at")
            values = observation.to_dict()
            values["posterior"] = tuple(values["posterior"])
            values["prior"] = tuple(values["prior"])
            values["tags"] = tuple(values["tags"])
            values["revoked_at"] = float(revoked_at)
            self._observations[observation_id] = MemoryObservation(**values)

    def retrieve(self, query: MemoryQuery) -> list[MemoryObservation]:
        with self._lock:
            values = [
                observation
                for observation in self._observations.values()
                if _is_active(observation, query)
            ]
        return list(
            select_budgeted_observations(
                values,
                nodes=query.nodes,
                root=query.root,
                limit=query.max_observations,
                policy=query.retrieval_policy,
            )
        )

    def retrieve_neighborhood(
        self, query: MemoryNeighborhoodQuery
    ) -> MemoryNeighborhood:
        with self._lock:
            values = list(self._observations.values())
        return _bounded_neighborhood(values, query)


class SQLiteMemoryStore:
    """Portable SQLite store; observations are immutable and revocation is timed."""

    _initialization_lock = threading.RLock()
    _journal_mode_retry_seconds = 30.0

    def __init__(self, path: str | Path, *, initialize: bool = True):
        self.path = Path(path)
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._lock = threading.RLock()
        if initialize:
            with self._initialization_lock:
                self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _snapshot(self):
        """Hold one consistent read snapshot across multiple SELECTs."""
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            migrate_schema(connection)
        self._enable_wal()

    def _enable_wal(self) -> None:
        """Enable WAL despite simultaneous first-open attempts by other processes."""
        deadline = time.monotonic() + self._journal_mode_retry_seconds
        delay = 0.005
        while True:
            try:
                with self._connection() as connection:
                    row = connection.execute(
                        "PRAGMA journal_mode=WAL"
                    ).fetchone()
                mode = "" if row is None else str(row[0]).lower()
                if mode != "wal":
                    raise RuntimeError(
                        f"SQLite refused WAL journal mode (reported {mode!r})"
                    )
                return
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", None)
                base_code = None if code is None else int(code) & 0xFF
                retryable = base_code in {
                    sqlite3.SQLITE_BUSY,
                    sqlite3.SQLITE_LOCKED,
                }
                if code is None:
                    message = str(error).lower()
                    retryable = (
                        "database is locked" in message
                        or "database table is locked" in message
                    )
                if not retryable or time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2.0, 0.1)

    def schema_status(self) -> SQLiteSchemaStatus:
        with self._lock, self._snapshot() as connection:
            return read_schema_status(connection)

    def check_schema(self) -> SQLiteSchemaStatus:
        with self._lock, self._snapshot() as connection:
            validate_current_schema(connection)
            return read_schema_status(connection)

    def monitor_config_payload(
        self, run_id: str = "default"
    ) -> dict[str, object] | None:
        with self._lock, self._snapshot() as connection:
            row = connection.execute(
                """
                SELECT config_json FROM monitor_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return None if row is None else dict(json.loads(row["config_json"]))

    def verify_integrity(self) -> dict[str, object]:
        """Read and validate the durable artifact without repairing it."""
        checks: dict[str, object] = {}
        errors: list[str] = []
        warnings: list[str] = []
        with self._lock, self._snapshot() as connection:
            quick = [
                str(row[0]) for row in connection.execute("PRAGMA quick_check")
            ]
            checks["quick_check"] = quick
            if quick != ["ok"]:
                errors.append("SQLite quick_check failed")
            foreign_keys = [
                tuple(row)
                for row in connection.execute("PRAGMA foreign_key_check")
            ]
            checks["foreign_key_violations"] = len(foreign_keys)
            if foreign_keys:
                errors.append("foreign-key violations were found")
            status = read_schema_status(connection)
            checks["schema_version"] = status.version
            try:
                validate_current_schema(connection)
                checks["schema_compatible"] = True
            except Exception as error:
                checks["schema_compatible"] = False
                errors.append(f"schema: {error}")
            if not checks["schema_compatible"]:
                return {
                    "ok": False,
                    "checks": checks,
                    "errors": errors,
                    "warnings": warnings,
                }
            prediction_rows = tuple(
                connection.execute(
                    """
                    SELECT
                        p.*,
                        o.namespace AS memory_namespace,
                        o.relation_type AS memory_relation_type,
                        o.posterior_json AS memory_posterior_json,
                        o.observed_at AS memory_observed_at
                    FROM monitor_predictions AS p
                    LEFT JOIN observations AS o
                      ON o.observation_id = p.observation_id
                    ORDER BY p.observation_id
                    """
                )
            )
            outcome_rows = tuple(
                connection.execute(
                    """
                    SELECT o.*, p.run_id
                    FROM monitor_outcomes AS o
                    LEFT JOIN monitor_predictions AS p
                      ON p.observation_id = o.observation_id
                    ORDER BY o.outcome_seq
                    """
                )
            )
            run_rows = tuple(
                connection.execute(
                    """
                    SELECT * FROM monitor_runs ORDER BY run_id
                    """
                )
            )
            law_rows = tuple(
                connection.execute(
                    """
                    SELECT * FROM monitor_laws
                    ORDER BY run_id, relation_type
                    """
                )
            )
            relation_law_rows = tuple(
                connection.execute(
                    """
                    SELECT * FROM relation_laws ORDER BY relation_type
                    """
                )
            )
            observation_rows = tuple(
                connection.execute(
                    """
                    SELECT * FROM observations ORDER BY observation_id
                    """
                )
            )
            checks["observations"] = len(observation_rows)
            checks["monitor_runs"] = len(run_rows)
            checks["relation_laws"] = len(relation_law_rows)
            checks["monitor_laws"] = len(law_rows)
            checks["monitor_predictions"] = len(prediction_rows)
            checks["monitor_outcomes"] = len(outcome_rows)
            unbound_relation_types = [
                str(row["relation_type"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT observation.relation_type
                    FROM observations AS observation
                    LEFT JOIN relation_laws AS law
                      ON law.relation_type = observation.relation_type
                    WHERE law.relation_type IS NULL
                    ORDER BY observation.relation_type
                    """
                )
            ]
            checks["unbound_observation_relation_types"] = (
                unbound_relation_types
            )
            if unbound_relation_types:
                errors.append(
                    "observations use relation types without a durable law: "
                    + ", ".join(repr(value) for value in unbound_relation_types)
                )
            monitors: dict[str, CalibrationDriftMonitor] = {}
            for row in run_rows:
                run_id = str(row["run_id"])
                try:
                    config_values = dict(json.loads(row["config_json"]))
                    monitor = CalibrationDriftMonitor(
                        MonitorConfig(run_id=run_id, **config_values)
                    )
                    if (
                        monitor.config.canonical_json
                        != str(row["config_json"])
                        or monitor.config.fingerprint
                        != str(row["config_sha256"])
                    ):
                        raise ValueError("monitor config hash mismatch")
                    monitors[run_id] = monitor
                except Exception as error:
                    errors.append(
                        f"monitor run {run_id!r}: {error}"
                    )
            laws: dict[tuple[str, str], tuple[list[str], str]] = {}
            global_laws: dict[str, tuple[list[str], str]] = {}
            for row in relation_law_rows:
                relation_type = str(row["relation_type"])
                try:
                    labels = json.loads(row["support_labels_json"])
                    if (
                        not isinstance(labels, list)
                        or not labels
                        or any(
                            not isinstance(label, str) or not label
                            for label in labels
                        )
                        or len(labels) != len(set(labels))
                    ):
                        raise ValueError("invalid support label snapshot")
                    fingerprint = str(row["law_fingerprint"])
                    if (
                        len(fingerprint) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in fingerprint
                        )
                    ):
                        raise ValueError("invalid law fingerprint")
                    global_laws[relation_type] = (labels, fingerprint)
                except Exception as error:
                    errors.append(
                        f"relation law {relation_type!r}: {error}"
                    )
            decoded_observations: dict[str, MemoryObservation] = {}
            for row in observation_rows:
                observation_id = str(row["observation_id"])
                try:
                    observation = self._decode(row)
                    decoded_observations[observation_id] = observation
                    law = global_laws.get(observation.relation_type)
                    if law is None:
                        continue
                    expected_size = len(law[0])
                    if (
                        len(observation.posterior) != expected_size
                        or len(observation.prior) != expected_size
                    ):
                        raise ValueError(
                            "posterior/prior support size differs from the "
                            "durable relation law"
                        )
                except Exception as error:
                    errors.append(
                        f"observation {observation_id!r}: {error}"
                    )
            for observation in decoded_observations.values():
                for parent_id in observation.derived_from:
                    parent = decoded_observations.get(parent_id)
                    if parent is None:
                        errors.append(
                            f"observation {observation.observation_id!r}: "
                            f"derived_from parent {parent_id!r} is missing"
                        )
                        continue
                    try:
                        _validate_lineage_parent(observation, parent)
                    except Exception as error:
                        errors.append(
                            f"observation {observation.observation_id!r}: "
                            f"{error}"
                        )
            for row in law_rows:
                key = (str(row["run_id"]), str(row["relation_type"]))
                try:
                    labels = json.loads(row["support_labels_json"])
                    if (
                        not isinstance(labels, list)
                        or not labels
                        or any(
                            not isinstance(label, str) or not label
                            for label in labels
                        )
                        or len(labels) != len(set(labels))
                    ):
                        raise ValueError("invalid support label snapshot")
                    fingerprint = str(row["law_fingerprint"])
                    if len(fingerprint) != 64:
                        raise ValueError("invalid law fingerprint")
                    if global_laws[str(row["relation_type"])] != (
                        labels,
                        fingerprint,
                    ):
                        raise ValueError(
                            "monitor law differs from global relation law"
                        )
                    laws[key] = (labels, fingerprint)
                except Exception as error:
                    errors.append(f"monitor law {key!r}: {error}")
            for row in prediction_rows:
                try:
                    prediction = self._decode_prediction(row)
                    law = laws[
                        (prediction.run_id, prediction.relation_type)
                    ]
                    if (
                        list(prediction.support_labels) != law[0]
                        or prediction.law_fingerprint != law[1]
                    ):
                        raise ValueError(
                            "prediction does not match registered monitor law"
                        )
                    if int(row["support_size"]) != len(
                        prediction.posterior
                    ):
                        raise ValueError(
                            "support_size differs from prediction payload"
                        )
                    if row["memory_namespace"] is None:
                        raise ValueError("observation snapshot is missing")
                    if (
                        prediction.namespace != str(row["memory_namespace"])
                        or prediction.relation_type
                        != str(row["memory_relation_type"])
                        or prediction.posterior
                        != tuple(
                            float(value)
                            for value in json.loads(
                                row["memory_posterior_json"]
                            )
                        )
                        or prediction.observed_at
                        != float(row["memory_observed_at"])
                    ):
                        raise ValueError(
                            "prediction differs from observation snapshot"
                        )
                    monitors[prediction.run_id].restore_prediction(prediction)
                except Exception as error:
                    errors.append(
                        f"prediction {row['observation_id']!r}: {error}"
                    )
            for row in outcome_rows:
                try:
                    outcome = self._decode_outcome(row)
                    run_id = row["run_id"]
                    if run_id is None:
                        raise ValueError("prediction run is missing")
                    monitors[str(run_id)].restore_outcome(outcome)
                except Exception as error:
                    errors.append(
                        f"outcome {row['outcome_event_id']!r}: {error}"
                    )
            root_counts = tuple(
                connection.execute(
                    """
                    SELECT observation_id, COUNT(*) AS count
                    FROM monitor_outcomes
                    WHERE supersedes_event_id IS NULL
                    GROUP BY observation_id
                    HAVING COUNT(*) > 1
                    """
                )
            )
            checks["outcome_root_conflicts"] = len(root_counts)
            if root_counts:
                errors.append("multiple root outcomes exist for an observation")
        return {
            "ok": not errors,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def _observation_parameters(
        observation: MemoryObservation,
    ) -> dict[str, object]:
        row = observation.to_dict()
        return {
            **row,
            "posterior_json": json.dumps(
                row["posterior"], separators=(",", ":"), allow_nan=False
            ),
            "prior_json": json.dumps(
                row["prior"], separators=(",", ":"), allow_nan=False
            ),
            "provenance_json": json.dumps(
                row["provenance"],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "independent_evidence": int(row["independent_evidence"]),
            "tags_json": json.dumps(
                row["tags"],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "derived_from_json": json.dumps(
                row["derived_from"],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }

    @staticmethod
    def _insert_observation(
        connection: sqlite3.Connection,
        observation: MemoryObservation,
    ) -> None:
        if observation.derived_from:
            placeholders = ",".join(
                "?" for _ in observation.derived_from
            )
            parent_rows = connection.execute(
                f"""
                SELECT * FROM observations
                WHERE observation_id IN ({placeholders})
                """,
                observation.derived_from,
            )
            parents = {
                str(row["observation_id"]): SQLiteMemoryStore._decode(row)
                for row in parent_rows
            }
            for parent_id in observation.derived_from:
                parent = parents.get(parent_id)
                if parent is None:
                    raise ValueError(
                        f"derived_from parent {parent_id!r} does not exist"
                    )
                _validate_lineage_parent(observation, parent)
        law_row = connection.execute(
            """
            SELECT support_labels_json
            FROM relation_laws
            WHERE relation_type = ?
            """,
            (observation.relation_type,),
        ).fetchone()
        if law_row is not None:
            try:
                labels = json.loads(law_row["support_labels_json"])
            except Exception as error:
                raise MonitorConfigurationConflictError(
                    "durable relation-law snapshot is malformed"
                ) from error
            if (
                not isinstance(labels, list)
                or not labels
                or len(observation.posterior) != len(labels)
                or len(observation.prior) != len(labels)
            ):
                raise ValueError(
                    "observation posterior/prior support does not match the "
                    "database-global durable relation law"
                )
        connection.execute(
            """
            INSERT INTO observations (
                observation_id, namespace, relation_type, source, target,
                posterior_json, prior_json, evidence_family,
                provenance_json, observed_at, valid_from, valid_until,
                independent_evidence, trust, tags_json, revoked_at,
                source_event_id, lineage_root_id, derived_from_json,
                encoder_revision, calibration_revision
            ) VALUES (
                :observation_id, :namespace, :relation_type, :source, :target,
                :posterior_json, :prior_json, :evidence_family,
                :provenance_json, :observed_at, :valid_from, :valid_until,
                :independent_evidence, :trust, :tags_json, :revoked_at,
                :source_event_id, :lineage_root_id, :derived_from_json,
                :encoder_revision, :calibration_revision
            )
            """,
            SQLiteMemoryStore._observation_parameters(observation),
        )

    def append(self, observation: MemoryObservation) -> None:
        with self._lock, self._transaction() as connection:
            try:
                self._insert_observation(connection, observation)
            except sqlite3.IntegrityError as error:
                raise DuplicateObservationError(
                    observation.observation_id
                ) from error

    @staticmethod
    def _validate_prediction_pair(
        observation: MemoryObservation,
        prediction: MonitorPredictionRecord,
    ) -> None:
        prediction.verify_hash()
        if not observation.independent_evidence:
            raise ValueError(
                "derived observations cannot create monitor predictions"
            )
        if (
            prediction.observation_id != observation.observation_id
            or prediction.namespace != observation.namespace
            or prediction.relation_type != observation.relation_type
            or prediction.posterior != observation.posterior
            or prediction.observed_at != observation.observed_at
        ):
            raise ValueError(
                "monitor prediction does not match its observation snapshot"
            )

    @staticmethod
    def _insert_prediction(
        connection: sqlite3.Connection,
        prediction: MonitorPredictionRecord,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO monitor_predictions (
                observation_id, run_id, namespace, relation_type,
                source_family, posterior_json, support_labels_json,
                support_size, law_fingerprint, observed_at, tracked_at,
                payload_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction.observation_id,
                prediction.run_id,
                prediction.namespace,
                prediction.relation_type,
                prediction.source_family,
                json.dumps(
                    prediction.posterior,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                json.dumps(
                    prediction.support_labels,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                len(prediction.posterior),
                prediction.law_fingerprint,
                prediction.observed_at,
                prediction.tracked_at,
                prediction.payload_sha256,
            ),
        )
        return int(cursor.lastrowid)

    def append_monitored(
        self,
        observation: MemoryObservation,
        prediction: MonitorPredictionRecord,
    ) -> MonitorPredictionRecord:
        """Atomically append an observation and its monitor snapshot."""
        self._validate_prediction_pair(observation, prediction)
        with self._lock, self._transaction() as connection:
            run = connection.execute(
                """
                SELECT config_json, config_sha256
                FROM monitor_runs WHERE run_id = ?
                """,
                (prediction.run_id,),
            ).fetchone()
            if run is None:
                raise MonitorConfigurationConflictError(
                    f"monitor run {prediction.run_id!r} is not registered"
                )
            config_values = json.loads(run["config_json"])
            if (
                not isinstance(config_values, dict)
                or "max_tracked" not in config_values
            ):
                raise MonitorConfigurationConflictError(
                    "durable monitor config is malformed"
                )
            tracked_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM monitor_predictions
                    WHERE run_id = ?
                    """,
                    (prediction.run_id,),
                ).fetchone()[0]
            )
            if tracked_count >= int(config_values["max_tracked"]):
                raise RuntimeError(
                    "durable monitor capacity reached; start a new locked run"
                )
            existing = connection.execute(
                """
                SELECT 1 FROM observations WHERE observation_id = ?
                """,
                (observation.observation_id,),
            ).fetchone()
            if existing is not None:
                raise DuplicateObservationError(observation.observation_id)
            try:
                self._insert_observation(connection, observation)
                prediction_seq = self._insert_prediction(
                    connection, prediction
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "monitor prediction transaction violated schema constraints"
                ) from error
            return replace(prediction, prediction_seq=prediction_seq)

    def bind_monitor(self, monitor: CalibrationDriftMonitor) -> None:
        """Hash-lock monitor configuration and replay its durable journal."""
        if (
            monitor.tracked_count != 0
            or monitor.prediction_high_water != 0
            or monitor.outcome_high_water != 0
        ):
            raise MonitorConfigurationConflictError(
                "a durable monitor must be empty before binding"
            )
        config = monitor.config
        created_at = time.time()
        with self._lock, self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT config_json, config_sha256
                FROM monitor_runs WHERE run_id = ?
                """,
                (config.run_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO monitor_runs(
                        run_id, config_json, config_sha256, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        config.run_id,
                        config.canonical_json,
                        config.fingerprint,
                        created_at,
                    ),
                )
            elif (
                str(existing["config_sha256"]) != config.fingerprint
                or str(existing["config_json"]) != config.canonical_json
            ):
                raise MonitorConfigurationConflictError(
                    f"monitor run {config.run_id!r} was created with a "
                    "different locked configuration"
                )
        self.sync_monitor(monitor)

    def sync_monitor(self, monitor: CalibrationDriftMonitor) -> None:
        """Replay newly visible durable events into a long-lived worker."""
        config = monitor.config
        prediction_high_water = monitor.prediction_high_water
        outcome_high_water = monitor.outcome_high_water
        with self._lock, self._snapshot() as connection:
            existing = connection.execute(
                """
                SELECT config_json, config_sha256
                FROM monitor_runs WHERE run_id = ?
                """,
                (config.run_id,),
            ).fetchone()
            if existing is None:
                raise MonitorConfigurationConflictError(
                    f"monitor run {config.run_id!r} does not exist"
                )
            if (
                str(existing["config_sha256"]) != config.fingerprint
                or str(existing["config_json"]) != config.canonical_json
            ):
                raise MonitorConfigurationConflictError(
                    f"monitor run {config.run_id!r} has a different "
                    "locked configuration"
                )
            prediction_rows = tuple(
                connection.execute(
                    """
                    SELECT * FROM monitor_predictions
                    WHERE run_id = ? AND prediction_seq > ?
                    ORDER BY prediction_seq
                    """,
                    (config.run_id, prediction_high_water),
                )
            )
            outcome_rows = tuple(
                connection.execute(
                    """
                    SELECT o.*
                    FROM monitor_outcomes AS o
                    JOIN monitor_predictions AS p
                      ON p.observation_id = o.observation_id
                    WHERE p.run_id = ?
                      AND o.outcome_seq > ?
                    ORDER BY o.outcome_seq
                    """,
                    (config.run_id, outcome_high_water),
                )
            )
            global_prediction_high_water = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(prediction_seq), 0)
                    FROM monitor_predictions
                    """
                ).fetchone()[0]
            )
            global_outcome_high_water = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(outcome_seq), 0)
                    FROM monitor_outcomes
                    """
                ).fetchone()[0]
            )
        for row in prediction_rows:
            monitor.restore_prediction(self._decode_prediction(row))
        for row in outcome_rows:
            monitor.restore_outcome(self._decode_outcome(row))
        monitor.advance_journal_cursors(
            global_prediction_high_water,
            global_outcome_high_water,
        )

    def assert_monitor_law(
        self,
        relation_type: str,
        support_labels: tuple[str, ...],
        law_fingerprint: str,
        *,
        run_id: str,
        law_payload: Mapping[str, object],
    ) -> None:
        relation_type, support_labels, law_fingerprint = (
            validate_relation_law_snapshot(
                relation_type,
                support_labels,
                law_fingerprint,
            )
        )
        self.bind_relation_law(
            relation_type,
            support_labels,
            law_fingerprint,
            law_payload=law_payload,
        )
        labels_json = json.dumps(
            support_labels,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._transaction() as connection:
            global_rows = tuple(
                connection.execute(
                    """
                    SELECT support_labels_json, law_fingerprint
                    FROM monitor_laws
                    WHERE relation_type = ?
                    """,
                    (relation_type,),
                )
            )
            if any(
                str(candidate["support_labels_json"]) != labels_json
                or str(candidate["law_fingerprint"]) != law_fingerprint
                for candidate in global_rows
            ):
                raise MonitorConfigurationConflictError(
                    f"relation law for {relation_type!r} differs from "
                    "another monitor run in this database"
                )
            row = connection.execute(
                """
                SELECT support_labels_json, law_fingerprint
                FROM monitor_laws
                WHERE run_id = ? AND relation_type = ?
                """,
                (run_id, relation_type),
            ).fetchone()
            if row is None:
                try:
                    connection.execute(
                        """
                        INSERT INTO monitor_laws(
                            run_id, relation_type, support_labels_json,
                            law_fingerprint, registered_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            relation_type,
                            labels_json,
                            law_fingerprint,
                            time.time(),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise MonitorConfigurationConflictError(
                        "monitor relation law could not be registered"
                    ) from error
                return
            if (
                str(row["support_labels_json"]) != labels_json
                or str(row["law_fingerprint"]) != law_fingerprint
            ):
                raise MonitorConfigurationConflictError(
                    f"registered law for {relation_type!r} does not match "
                    "the durable monitor snapshot"
                )

    def bind_relation_law(
        self,
        relation_type: str,
        support_labels: tuple[str, ...],
        law_fingerprint: str,
        *,
        law_payload: Mapping[str, object],
    ) -> None:
        """Atomically establish one database-global law per relation type.

        A complete validated law payload is required so a caller cannot lock
        an arbitrary digest without demonstrating its canonical preimage.
        """
        from .laws import FiniteRelationLaw
        from .monitoring import relation_law_fingerprint

        relation_type, support_labels, law_fingerprint = (
            validate_relation_law_snapshot(
                relation_type,
                support_labels,
                law_fingerprint,
            )
        )
        if not isinstance(law_payload, Mapping):
            raise ValueError("law_payload must be a complete law mapping")
        validated_law = FiniteRelationLaw.from_dict(dict(law_payload))
        if validated_law.labels != support_labels:
            raise ValueError(
                "law_payload labels do not match support_labels"
            )
        if relation_law_fingerprint(validated_law.to_dict()) != law_fingerprint:
            raise ValueError(
                "law_fingerprint does not match the canonical law payload"
            )
        labels_json = json.dumps(
            support_labels,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                """
                SELECT support_labels_json, law_fingerprint
                FROM relation_laws
                WHERE relation_type = ?
                """,
                (relation_type,),
            ).fetchone()
            if row is None:
                incompatible = connection.execute(
                    """
                    SELECT observation_id
                    FROM observations
                    WHERE relation_type = ?
                      AND (
                          CASE
                              WHEN json_valid(posterior_json)
                               AND json_type(posterior_json) = 'array'
                              THEN json_array_length(posterior_json)
                              ELSE -1
                          END <> ?
                          OR
                          CASE
                              WHEN json_valid(prior_json)
                               AND json_type(prior_json) = 'array'
                              THEN json_array_length(prior_json)
                              ELSE -1
                          END <> ?
                      )
                    ORDER BY observation_id
                    LIMIT 1
                    """,
                    (
                        relation_type,
                        len(support_labels),
                        len(support_labels),
                    ),
                ).fetchone()
                if incompatible is not None:
                    raise MonitorConfigurationConflictError(
                        f"cannot bind relation law {relation_type!r}: "
                        f"legacy observation "
                        f"{str(incompatible['observation_id'])!r} has a "
                        "different posterior/prior support size"
                    )
                for legacy_row in connection.execute(
                    """
                    SELECT * FROM observations
                    WHERE relation_type = ?
                    ORDER BY observation_id
                    """,
                    (relation_type,),
                ):
                    try:
                        legacy_observation = self._decode(legacy_row)
                    except Exception as error:
                        raise MonitorConfigurationConflictError(
                            f"cannot bind relation law {relation_type!r}: "
                            f"legacy observation "
                            f"{str(legacy_row['observation_id'])!r} is invalid"
                        ) from error
                    if (
                        len(legacy_observation.posterior)
                        != len(support_labels)
                        or len(legacy_observation.prior)
                        != len(support_labels)
                    ):
                        raise MonitorConfigurationConflictError(
                            f"cannot bind relation law {relation_type!r}: "
                            f"legacy observation "
                            f"{legacy_observation.observation_id!r} has a "
                            "different posterior/prior support size"
                        )
                try:
                    connection.execute(
                        """
                        INSERT INTO relation_laws(
                            relation_type, support_labels_json,
                            law_fingerprint, registered_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            relation_type,
                            labels_json,
                            law_fingerprint,
                            time.time(),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise MonitorConfigurationConflictError(
                        "durable relation law could not be registered"
                    ) from error
                return
            if (
                str(row["support_labels_json"]) != labels_json
                or str(row["law_fingerprint"]) != law_fingerprint
            ):
                raise MonitorConfigurationConflictError(
                    f"relation law for {relation_type!r} differs from the "
                    "database-global durable registry"
                )

    def check_relation_law(
        self,
        relation_type: str,
        support_labels: tuple[str, ...],
        law_fingerprint: str,
        *,
        require_registered: bool = True,
    ) -> None:
        """Read-only validation of the database-global relation registry.

        Relation types are database-global because observations are not
        partitioned by monitor run. An empty database may be queried with a
        caller-supplied law without mutating it, but existing observations must
        never be interpreted under an unbound law.
        """
        relation_type, support_labels, law_fingerprint = (
            validate_relation_law_snapshot(
                relation_type,
                support_labels,
                law_fingerprint,
            )
        )
        labels_json = json.dumps(
            support_labels,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._snapshot() as connection:
            rows = tuple(
                connection.execute(
                    """
                    SELECT support_labels_json, law_fingerprint
                    FROM relation_laws
                    WHERE relation_type = ?
                    """,
                    (relation_type,),
                )
            )
            observation_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM observations
                    WHERE relation_type = ?
                    """,
                    (relation_type,),
                ).fetchone()[0]
            )
        if not rows:
            if require_registered and observation_count:
                raise MonitorConfigurationConflictError(
                    f"relation type {relation_type!r} has observations but "
                    "no database-global durable law"
                )
            return
        if (
            str(rows[0]["support_labels_json"]) != labels_json
            or str(rows[0]["law_fingerprint"]) != law_fingerprint
        ):
            raise MonitorConfigurationConflictError(
                f"query law for {relation_type!r} does not match the "
                "database-global durable relation registry"
            )

    def check_monitor_law(
        self,
        relation_type: str,
        support_labels: tuple[str, ...],
        law_fingerprint: str,
        *,
        run_id: str,
    ) -> None:
        """Compatibility alias for the pre-v0.8 internal method name."""
        del run_id
        self.check_relation_law(
            relation_type,
            support_labels,
            law_fingerprint,
        )

    @staticmethod
    def _decode_prediction(row: sqlite3.Row) -> MonitorPredictionRecord:
        return MonitorPredictionRecord(
            observation_id=str(row["observation_id"]),
            run_id=str(row["run_id"]),
            namespace=str(row["namespace"]),
            relation_type=str(row["relation_type"]),
            source_family=str(row["source_family"]),
            posterior=tuple(float(value) for value in json.loads(row["posterior_json"])),
            support_labels=tuple(
                str(value) for value in json.loads(row["support_labels_json"])
            ),
            law_fingerprint=str(row["law_fingerprint"]),
            observed_at=float(row["observed_at"]),
            tracked_at=float(row["tracked_at"]),
            payload_sha256=str(row["payload_sha256"]),
            prediction_seq=int(row["prediction_seq"]),
        )

    @staticmethod
    def _decode_outcome(row: sqlite3.Row) -> MonitorOutcomeRecord:
        return MonitorOutcomeRecord(
            outcome_event_id=str(row["outcome_event_id"]),
            observation_id=str(row["observation_id"]),
            relation_type=str(row["relation_type"]),
            true_index=int(row["true_index"]),
            true_label=str(row["true_label"]),
            law_fingerprint=str(row["law_fingerprint"]),
            labeled_at=float(row["labeled_at"]),
            recorded_at=float(row["recorded_at"]),
            supersedes_event_id=(
                None
                if row["supersedes_event_id"] is None
                else str(row["supersedes_event_id"])
            ),
            correction_reason=(
                None
                if row["correction_reason"] is None
                else str(row["correction_reason"])
            ),
            payload_sha256=str(row["payload_sha256"]),
            outcome_seq=int(row["outcome_seq"]),
        )

    def append_monitor_outcome(
        self, event: MonitorOutcomeRecord
    ) -> tuple[MonitorOutcomeRecord, bool]:
        """Append one outcome/correction event with strict idempotency."""
        event.verify_hash()
        with self._lock, self._transaction() as connection:
            existing_by_id = connection.execute(
                """
                SELECT * FROM monitor_outcomes
                WHERE outcome_event_id = ?
                """,
                (event.outcome_event_id,),
            ).fetchone()
            if existing_by_id is not None:
                existing = self._decode_outcome(existing_by_id)
                if existing.payload_sha256 == event.payload_sha256:
                    return existing, False
                raise MonitorIdempotencyConflictError(
                    f"outcome event id {event.outcome_event_id!r} was reused "
                    "with a different payload"
                )
            prediction = connection.execute(
                """
                SELECT * FROM monitor_predictions WHERE observation_id = ?
                """,
                (event.observation_id,),
            ).fetchone()
            if prediction is None:
                raise MonitorObservationNotTrackedError(event.observation_id)
            current_row = connection.execute(
                """
                SELECT candidate.*
                FROM monitor_outcomes AS candidate
                WHERE candidate.observation_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM monitor_outcomes AS child
                      WHERE child.supersedes_event_id =
                            candidate.outcome_event_id
                  )
                """,
                (event.observation_id,),
            ).fetchone()
            current_id = (
                None
                if current_row is None
                else str(current_row["outcome_event_id"])
            )
            if event.supersedes_event_id != current_id:
                raise MonitorOutcomeConflictError(
                    "outcome does not extend the durable correction head"
                )
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO monitor_outcomes (
                        outcome_event_id, observation_id, relation_type,
                        true_index, true_label, law_fingerprint, labeled_at,
                        recorded_at, supersedes_event_id, correction_reason,
                        payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.outcome_event_id,
                        event.observation_id,
                        event.relation_type,
                        event.true_index,
                        event.true_label,
                        event.law_fingerprint,
                        event.labeled_at,
                        event.recorded_at,
                        event.supersedes_event_id,
                        event.correction_reason,
                        event.payload_sha256,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise MonitorOutcomeConflictError(
                    "outcome violates the append-only journal contract"
                ) from error
            return replace(event, outcome_seq=int(cursor.lastrowid)), True

    def revoke(self, observation_id: str, revoked_at: float) -> None:
        timestamp = float(revoked_at)
        if not math.isfinite(timestamp):
            raise ValueError("revoked_at must be finite")
        with self._lock, self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE observations SET revoked_at = ?
                WHERE observation_id = ?
                  AND revoked_at IS NULL
                  AND observed_at <= ?
                """,
                (timestamp, observation_id, timestamp),
            )
            if cursor.rowcount == 1:
                return
            row = connection.execute(
                """
                SELECT observed_at, revoked_at FROM observations
                WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            if row is not None and row["revoked_at"] is None:
                raise ValueError("revoked_at must not precede observed_at")
            raise ObservationNotFoundError(observation_id)

    @staticmethod
    def _decode(row: sqlite3.Row) -> MemoryObservation:
        return MemoryObservation(
            observation_id=row["observation_id"],
            namespace=row["namespace"],
            relation_type=row["relation_type"],
            source=row["source"],
            target=row["target"],
            posterior=tuple(json.loads(row["posterior_json"])),
            prior=tuple(json.loads(row["prior_json"])),
            evidence_family=row["evidence_family"],
            provenance=json.loads(row["provenance_json"]),
            observed_at=float(row["observed_at"]),
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            independent_evidence=bool(row["independent_evidence"]),
            trust=float(row["trust"]),
            tags=tuple(json.loads(row["tags_json"])),
            revoked_at=row["revoked_at"],
            source_event_id=row["source_event_id"],
            lineage_root_id=row["lineage_root_id"],
            derived_from=tuple(json.loads(row["derived_from_json"])),
            encoder_revision=row["encoder_revision"],
            calibration_revision=row["calibration_revision"],
        )

    def retrieve(self, query: MemoryQuery) -> list[MemoryObservation]:
        placeholders = ",".join("?" for _ in query.nodes)
        parameters = [
            query.namespace,
            query.relation_type,
            query.as_of,
            query.as_of,
            query.as_of,
            query.as_of,
            *query.nodes,
            *query.nodes,
        ]
        sql = f"""
            SELECT * FROM observations
            WHERE namespace = ?
              AND relation_type = ?
              AND (revoked_at IS NULL OR revoked_at > ?)
              AND (valid_from IS NULL OR valid_from <= ?)
              AND (valid_until IS NULL OR valid_until >= ?)
              AND observed_at <= ?
              AND source IN ({placeholders})
              AND target IN ({placeholders})
            ORDER BY observed_at DESC, observation_id ASC
        """
        with self._lock, self._connection() as connection:
            values = [self._decode(row) for row in connection.execute(sql, parameters)]
        if query.tags:
            required = set(query.tags)
            values = [value for value in values if required.issubset(value.tags)]
        return list(
            select_budgeted_observations(
                values,
                nodes=query.nodes,
                root=query.root,
                limit=query.max_observations,
                policy=query.retrieval_policy,
            )
        )

    def retrieve_neighborhood(
        self, query: MemoryNeighborhoodQuery
    ) -> MemoryNeighborhood:
        tag_clauses = []
        parameters = [
            query.namespace,
            query.relation_type,
            query.as_of,
            query.as_of,
            query.as_of,
            query.as_of,
        ]
        for tag in query.tags:
            tag_clauses.append(
                """
                AND EXISTS (
                    SELECT 1 FROM json_each(o.tags_json)
                    WHERE json_each.value = ?
                )
                """
            )
            parameters.append(tag)
        seed_values = ", ".join("(?, 0)" for _ in query.seeds)
        parameters.extend(query.seeds)
        parameters.extend((query.max_hops, query.max_hops))
        sql = f"""
            WITH RECURSIVE
            ranked AS (
                SELECT
                    o.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(
                            o.lineage_root_id,
                            o.source_event_id,
                            o.evidence_family
                        )
                        ORDER BY o.observed_at DESC, o.observation_id DESC
                    ) AS family_rank
                FROM observations AS o
                WHERE o.namespace = ?
                  AND o.relation_type = ?
                  AND (o.revoked_at IS NULL OR o.revoked_at > ?)
                  AND (o.valid_from IS NULL OR o.valid_from <= ?)
                  AND (o.valid_until IS NULL OR o.valid_until >= ?)
                  AND o.observed_at <= ?
                  AND o.independent_evidence = 1
                  {''.join(tag_clauses)}
            ),
            edges AS (
                SELECT * FROM ranked WHERE family_rank = 1
            ),
            seeds(node, depth) AS (
                VALUES {seed_values}
            ),
            reachable(node, depth) AS (
                SELECT node, depth FROM seeds
                UNION
                SELECT
                    CASE
                        WHEN edges.source = reachable.node THEN edges.target
                        ELSE edges.source
                    END,
                    reachable.depth + 1
                FROM reachable
                JOIN edges
                  ON edges.source = reachable.node
                  OR edges.target = reachable.node
                WHERE reachable.depth < ?
            )
            SELECT edges.*
            FROM edges
            WHERE EXISTS (
                SELECT 1
                FROM reachable
                WHERE reachable.depth < ?
                  AND (
                    reachable.node = edges.source
                    OR reachable.node = edges.target
                  )
            )
            ORDER BY edges.observed_at DESC, edges.observation_id ASC
        """
        with self._lock, self._connection() as connection:
            values = [self._decode(row) for row in connection.execute(sql, parameters)]
        return _bounded_neighborhood(values, query)
