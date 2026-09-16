# CLI Reference

All commands follow the shape:

```
posterior-memory --db <sqlite-path> [global flags] <command> [command args]
```

Output is JSON on stdout. Errors are JSON on stderr:
`{"error": {"code": "<kind>", "message": "..."}}` where kind is
`conflict`, `schema`, `input`, or `internal`.

## Global flags

| Flag | Required | Meaning |
|---|---|---|
| `--db PATH` | Always | SQLite database. Parent directory must exist. |
| `--relation-type NAME` | Depends (see below) | Relation type, e.g. `task-phase`. May be omitted if the payload embeds a matching `relation_type` (observe/query/decision only). |
| `--law SPEC` | observe/query/decision/outcome | `s3`, `cyclic:N`, or a path to a finite-law JSON file. |
| `--monitor-run ID` | No (default `default`) | Hash-locked durable monitor run. |
| `--monitor-config PATH` | Only when creating a run | JSON MonitorConfig; `run_id` in the file must equal `--monitor-run`. |

## Commands

### observe `<payload.json | ->`

Validates and stores one observation; binds the relation law durably on
first use; records a monitor prediction snapshot in the same transaction.

```bash
posterior-memory --db m.sqlite --relation-type task-phase --law cyclic:4 \
  observe observation.json
```

Output: `{"observation_id": "..."}`.

### query `<payload.json | ->`

Reconciles all matching active observations and returns the full capsule.
Exactly one of `nodes` (explicit) or `seeds` (neighborhood discovery) must
be present; supplying both or neither is rejected.

```bash
posterior-memory --db m.sqlite --relation-type task-phase --law cyclic:4 \
  query query.json
```

On an empty database this returns a prior-only capsule. If matching
observations exist but the law was never bound to this database, it fails
closed (exit 3).

### decision `<payload.json | ->` `--focus-node NODE` `[--alternatives K]`

Same query, but returns the compact `memory_decision` projection for one
focus node (`--alternatives` defaults to 2). Does not create a durable law
registration; requires a previously bound law when observations exist.

### outcome `<payload.json | ->`

Records delayed ground truth for an observation. Requires
`--relation-type` and `--law` flags and an existing monitor run.

```bash
posterior-memory --db m.sqlite --relation-type task-phase --law cyclic:4 \
  outcome outcome.json
```

To correct an earlier label, append a second outcome payload with
`"correction_reason": "..."` and a new `outcome_event_id`. Never edit the
first event. Output: `{"outcome_event_id", "observation_id", "recorded",
"correction"}`.

### health `[--namespace N] [--source-family F] [--as-of T]`

Calibration/drift report for the monitor run. No `--law` needed. The run
must already exist (created by the first `observe`); health never creates
or configures a run. With `--relation-type` or `--source-family` it
returns the segmented monitor report instead of the calibration summary.

Statuses: `healthy`, `drift`-style alerts, or `insufficient_data` (never
reported as healthy when the labeled window is too small).

### revoke `<observation_id>` `[--at T]`

Timed revocation; history is preserved (the record is not erased).
`--at` defaults to now. No law needed.

### verify

Read-only integrity check (quick_check, foreign keys, schema, journal
shape, payload hashes). Exit code 3 when `"ok"` is false. Never repairs.

## Law specification

- `--law cyclic:4` — cyclic group, labels `"0".."3"`.
- `--law s3` — noncommutative symmetric group on 3 states.
- `--law ./law.json` — custom finite law:

```json
{
  "name": "z2-parity",
  "labels": ["even", "odd"],
  "table": [[0, 1], [1, 0]],
  "inverse": [0, 1],
  "identity": 0
}
```

The constructor exhaustively validates closure, identity, inverses, and
associativity. Noncommutative laws are supported; reversed multiplication
order is a different law. A `relation_type` can never silently change its
law after binding.

## Monitor runs

- First `observe` creates the run: `default` with default thresholds, or
  a custom `--monitor-run <id> --monitor-config <json>`.
- The canonicalized config is SHA-256 locked by `run_id`; reopening with
  changed thresholds fails closed (exit 4).
- `health` and `outcome` require the run to exist.
- `max_tracked` is a hard capacity; roll to a new `run_id` instead of
  changing a full reference window.

MonitorConfig fields (all optional with defaults):
`reference_size` (200), `current_size` (100), `min_current_size` (30),
`ece_bins` (10), `max_nll_increase` (0.15), `max_brier_increase` (0.05),
`max_ece` (0.1), `max_prediction_js` (0.08), `max_tracked` (100000).

## Schema tool

```bash
posterior-memory-schema --db m.sqlite check     # read-only, creates nothing
posterior-memory-schema --db m.sqlite migrate   # explicit migration (default)
```

A database created by a newer package version is refused, never silently
downgraded. Migrations are idempotent and keep an append-only history.

## Exit codes

| Code | Kind | Meaning |
|---:|---|---|
| 0 | — | success |
| 1 | internal | unexpected failure |
| 2 | input | invalid input/payload/JSON |
| 3 | schema | incompatible schema or failed integrity check |
| 4 | conflict | durable state/idempotency/monitor conflict |

## Strict input rules

Payloads are validated before any storage: unknown fields, string
booleans, boolean probabilities, non-finite numbers, duplicate IDs,
oversized relation supports, and non-JSON provenance are rejected.
`evidence_family` is mandatory. Resource limits cover identifier length,
tag count, provenance bytes/depth, query nodes, hops, and observation
budgets. Payload `relation_type` must match `--relation-type` when both
are present.
