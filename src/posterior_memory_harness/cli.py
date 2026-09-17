"""JSON CLI for language- and framework-neutral integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .harness import PosteriorMemoryHarness
from .laws import FiniteRelationLaw, cyclic_law, s3_law
from .monitoring import (
    CalibrationDriftMonitor,
    MonitorConfig,
    relation_law_fingerprint,
)
from .projection import project_memory_decision
from .sqlite_schema import (
    SchemaMigrationError,
    UnsupportedSchemaVersionError,
)
from .store import (
    DuplicateObservationError,
    MonitorConfigurationConflictError,
    MonitorIdempotencyConflictError,
    MonitorObservationNotTrackedError,
    MonitorOutcomeConflictError,
    ObservationNotFoundError,
    SQLiteMemoryStore,
)
from .validation import (
    PayloadValidationError,
    validate_outcome_payload,
)


def read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_law(specification: str) -> FiniteRelationLaw:
    if specification == "s3":
        return s3_law()
    if specification.startswith("cyclic:"):
        return cyclic_law(int(specification.split(":", 1)[1]))
    payload = read_json(specification)
    if not isinstance(payload, dict):
        raise ValueError("finite-law JSON must be an object")
    return FiniteRelationLaw.from_dict(payload)


def _load_monitor(
    args: argparse.Namespace,
    store: SQLiteMemoryStore,
    *,
    require_existing: bool,
) -> CalibrationDriftMonitor:
    persisted = store.monitor_config_payload(args.monitor_run)
    if require_existing and persisted is None:
        raise MonitorConfigurationConflictError(
            f"monitor run {args.monitor_run!r} does not exist; create it "
            "with an observed prediction first"
        )
    if args.monitor_config is not None:
        payload = read_json(args.monitor_config)
        if not isinstance(payload, dict):
            raise ValueError("monitor config must be a JSON object")
        payload = dict(payload)
        configured_run = payload.pop("run_id", args.monitor_run)
        if configured_run != args.monitor_run:
            raise ValueError(
                "monitor config run_id must match --monitor-run"
            )
        try:
            config = MonitorConfig(run_id=args.monitor_run, **payload)
        except TypeError as error:
            raise ValueError(f"invalid monitor config: {error}") from error
    else:
        config = MonitorConfig(
            run_id=args.monitor_run,
            **({} if persisted is None else persisted),
        )
    return CalibrationDriftMonitor(config)


def _require_law_arguments(args: argparse.Namespace) -> None:
    if not args.relation_type:
        raise ValueError("--relation-type is required for this command")
    if not args.law:
        raise ValueError("--law is required for this command")


def build_harness(
    args: argparse.Namespace,
    *,
    store: SQLiteMemoryStore | None = None,
    require_law: bool = True,
    use_monitor: bool = True,
    require_existing_monitor: bool = False,
    persist_law: bool = True,
) -> PosteriorMemoryHarness:
    store = store or SQLiteMemoryStore(args.db)
    monitor = (
        _load_monitor(
            args,
            store,
            require_existing=require_existing_monitor,
        )
        if use_monitor
        else None
    )
    harness = PosteriorMemoryHarness(store, monitor=monitor)
    if require_law:
        _require_law_arguments(args)
        law = load_law(args.law)
        if persist_law:
            harness.register_relation(args.relation_type, law)
        else:
            store.check_relation_law(
                args.relation_type,
                law.labels,
                relation_law_fingerprint(law.to_dict()),
                require_registered=True,
            )
            harness.registry.register(args.relation_type, law)
    return harness


def _relation_payload(
    payload: Any, relation_type: str | None
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PayloadValidationError("payload must be an object")
    result = dict(payload)
    embedded = result.get("relation_type")
    if relation_type is None:
        if not isinstance(embedded, str) or not embedded:
            raise PayloadValidationError(
                "relation_type is required in the payload or CLI"
            )
    elif embedded is None:
        result["relation_type"] = relation_type
    elif embedded != relation_type:
        raise PayloadValidationError(
            "payload relation_type conflicts with --relation-type"
        )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="posterior-memory")
    parser.add_argument("--db", required=True, help="SQLite memory database")
    parser.add_argument("--relation-type")
    parser.add_argument(
        "--law",
        help="s3, cyclic:N, or a path to a finite-law JSON file",
    )
    parser.add_argument(
        "--monitor-run",
        default="default",
        help="hash-locked durable monitor run (default: default)",
    )
    parser.add_argument(
        "--monitor-config",
        help="JSON MonitorConfig; required only when creating/customizing a run",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    observe = subparsers.add_parser("observe")
    observe.add_argument("payload", help="observation JSON path, or - for stdin")

    query = subparsers.add_parser("query")
    query.add_argument("payload", help="query JSON path, or - for stdin")

    decision = subparsers.add_parser("decision")
    decision.add_argument("payload", help="query JSON path, or - for stdin")
    decision.add_argument("--focus-node", required=True)
    decision.add_argument("--alternatives", type=int, default=2)

    outcome = subparsers.add_parser("outcome")
    outcome.add_argument("payload", help="outcome JSON path, or - for stdin")

    health = subparsers.add_parser("health")
    health.add_argument("--namespace")
    health.add_argument("--source-family")
    health.add_argument("--as-of", type=float)

    revoke = subparsers.add_parser("revoke")
    revoke.add_argument("observation_id")
    revoke.add_argument("--at", type=float)

    subparsers.add_parser("verify")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "revoke":
        store = SQLiteMemoryStore(args.db)
        import time

        store.revoke(
            args.observation_id,
            time.time() if args.at is None else args.at,
        )
        return {"revoked": args.observation_id}

    if args.command == "verify":
        store = SQLiteMemoryStore(args.db, initialize=False)
        return store.verify_integrity()

    if args.command == "health":
        store = SQLiteMemoryStore(args.db)
        harness = build_harness(
            args,
            store=store,
            require_law=False,
            require_existing_monitor=True,
        )
        if args.relation_type is not None or args.source_family is not None:
            return harness.monitor.report(
                namespace=args.namespace,
                relation_type=args.relation_type,
                source_family=args.source_family,
                as_of=args.as_of,
            )
        return harness.calibration_report(
            namespace=args.namespace,
            as_of=args.as_of,
        )

    if args.command == "outcome":
        _require_law_arguments(args)
        harness = build_harness(args, require_existing_monitor=True)
        payload = validate_outcome_payload(read_json(args.payload))
        reason = payload["correction_reason"]
        if reason is None:
            recorded = harness.record_outcome(
                payload["observation_id"],
                payload["true_relation"],
                labeled_at=payload["labeled_at"],
                outcome_event_id=payload["outcome_event_id"],
            )
        else:
            recorded = harness.correct_outcome(
                payload["observation_id"],
                payload["true_relation"],
                reason=reason,
                labeled_at=payload["labeled_at"],
                outcome_event_id=payload["outcome_event_id"],
            )
        event = harness.monitor.current_outcome(payload["observation_id"])
        return {
            "outcome_event_id": (
                None if event is None else event.outcome_event_id
            ),
            "observation_id": payload["observation_id"],
            "recorded": recorded,
            "correction": reason is not None,
        }

    payload = _relation_payload(read_json(args.payload), args.relation_type)
    if args.relation_type is None:
        args.relation_type = payload["relation_type"]
    harness = build_harness(
        args,
        use_monitor=args.command == "observe",
        persist_law=args.command == "observe",
    )
    if args.command == "observe":
        return {"observation_id": harness.observe_payload(payload)}
    capsule = harness.query_payload(payload)
    if args.command == "query":
        return capsule
    return project_memory_decision(
        capsule,
        args.focus_node,
        alternatives=args.alternatives,
    )


def _error_code(error: Exception) -> tuple[int, str]:
    if isinstance(
        error,
        (
            MonitorConfigurationConflictError,
            MonitorIdempotencyConflictError,
            MonitorOutcomeConflictError,
            MonitorObservationNotTrackedError,
            ObservationNotFoundError,
            DuplicateObservationError,
        ),
    ):
        return 4, "conflict"
    if isinstance(
        error,
        (
            SchemaMigrationError,
            UnsupportedSchemaVersionError,
        ),
    ):
        return 3, "schema"
    if isinstance(
        error,
        (
            PayloadValidationError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            FileNotFoundError,
        ),
    ):
        return 2, "input"
    return 1, "internal"


def main(argv=None):
    args = parse_args(argv)
    try:
        result = _run(args)
    except Exception as error:
        code, kind = _error_code(error)
        print(
            json.dumps(
                {
                    "error": {
                        "code": kind,
                        "message": str(error),
                    }
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return code
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "verify" and not result.get("ok", False):
        return 3
    return 0


def schema_main(argv=None):
    parser = argparse.ArgumentParser(prog="posterior-memory-schema")
    parser.add_argument("--db", required=True, help="SQLite memory database")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("check", "migrate"),
        default="migrate",
        help="check is read-only; migrate is the backwards-compatible default",
    )
    args = parser.parse_args(argv)
    try:
        store = SQLiteMemoryStore(
            args.db,
            initialize=args.command == "migrate",
        )
        status = (
            store.check_schema()
            if args.command == "check"
            else store.schema_status()
        )
    except Exception as error:
        code, kind = _error_code(error)
        print(
            json.dumps(
                {"error": {"code": kind, "message": str(error)}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return code
    print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
