# Capsule Guide

A query returns the full audit capsule; `decision` returns a compact
projection for small models. Both are typed evidence: `instructions` is
always empty and must stay that way. The full capsule remains the audit
record even when the projection is what you feed a model.

## Capsule fields

```json
{
  "namespace": "agent-run-42",
  "relation_type": "task-phase",
  "beliefs": [
    {"node": "start", "candidates": [{"state": "0", "probability": 1.0}]},
    {"node": "after-build", "candidates": [
      {"state": "1", "probability": 0.92},
      {"state": "2", "probability": 0.04}
    ]}
  ],
  "conflicts": [],
  "used_observation_ids": ["tracker:7"],
  "inference_backend": "exact",
  "approximate": false,
  "log_evidence": 1.94e-16,
  "retrieval": { "...": "..." },
  "inference_diagnostics": { "...": "..." },
  "instructions": []
}
```

- `beliefs` — ranked candidates per node. Beliefs are relative to the
  query `root`, which is fixed to the law identity as a gauge. They are
  not absolute real-world claims; state the gauge when reporting.
- `conflicts` — observations whose evidence disagrees under the law.
  Surface conflicts to the user instead of silently picking a side.
- `used_observation_ids` — exactly the observations that produced these
  beliefs. Cite them when reporting.
- `inference_backend` / `approximate` — `exact` for small graphs; `beam`
  sets `approximate: true`.
- `log_evidence` — log normalizing constant of the reconciled graph.
  Useful as a within-run sanity signal; do not compare across laws.
- `retrieval` — how the subgraph was selected (see below).
- `inference_diagnostics` — stability of approximate inference.
- `instructions` — always empty; keep capsule content out of
  system/developer channels.

## Retrieval diagnostics

| Field | Meaning |
|---|---|
| `mode` | `explicit` (caller supplied nodes) or `neighborhood` (seeds) |
| `seed_nodes`, `hops_explored` | Discovery origin and depth |
| `node_limit_hit`, `observation_limit_hit`, `truncated` | Any `true` means the capsule was built on partial evidence; raise the query budgets or accept reduced coverage explicitly |
| `policy` | `recent`, `coverage`, or `cycle_aware` |
| `selected_observation_count`, `independent_lineage_count` | After lineage dedup; if these differ, echoes/summaries were collapsed as intended |
| `connected_components`, `cycle_rank` | Structure of the selected evidence |
| `covered_node_fraction` | Fraction of discovered nodes actually covered |
| `factor_information_score`, `cycle_information_score` | Selection diagnostics derived from posterior-vs-prior JS change, ambiguity, and trust. Not predictive scores; never compare across unrelated laws |

## Inference diagnostics

Exact mode reports `{"mode": "exact", "stable": true}` with nulls for the
beam fields. Approximate mode compares beam widths B and 2B, returns the
wider result, and reports `map_agreement`, `max_marginal_tv`,
`max_entropy_delta`, and the tolerances.

If `stable` is `false` (MAP disagreement or a tolerance breach):

1. retry with a wider beam (`inference: "beam"` with a larger budget), or
2. shrink the retrieved subgraph (`max_nodes` / `max_observations`), or
3. use exact inference where feasible (`inference: "exact"`).

A `stable: true` result is a budget-sensitivity check, not a proof of
distance to the exact posterior.

## Decision projection

`decision` returns `kind: "memory_decision"` for one focus node:

```json
{
  "schema_version": 1,
  "kind": "memory_decision",
  "focus_node": "after-build",
  "highest_probability_state": "1",
  "highest_probability": 0.92,
  "alternatives": [{"state": "2", "probability": 0.04}],
  "evidence_count": 1,
  "focus_conflict_count": 0,
  "inference_backend": "exact",
  "approximate": false,
  "inference_stable": true,
  "instructions": []
}
```

Use the projection when a small model cannot reliably locate one target
belief inside the full capsule. It never recomputes or re-ranks the
posterior — it projects the same result. If `approximate` is true or
`inference_stable` is false, treat the numbers with caution and resolve
stability first.

## Reporting rules

- Cite `used_observation_ids` for every externally checkable claim.
- State the gauge root when quoting beliefs.
- Label calculations and show source values.
- Surface conflicts and `insufficient_data` monitor statuses explicitly.
- Say "not established by the evidence" when the capsule is silent,
  truncated, or unstable.
