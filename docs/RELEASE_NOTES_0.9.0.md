# Posterior Memory Harness 0.9.0

## Release focus

Version 0.9 makes the retrieval budget structurally aware. It can preserve
independent constraint coverage and informative cycle-closing factors instead
of selecting observations only by recency.

## New

- Evidence lineage metadata: `source_event_id`, `lineage_root_id`,
  `derived_from`, `encoder_revision`, and `calibration_revision`.
- Cross-family deduplication by lineage root, with the previous
  `evidence_family` behavior retained as the fallback.
- Parent-before-child lineage validation; independent derived observations
  must preserve the parent lineage and relation endpoints.
- Three matched-budget retrieval policies: `recent`, `coverage`, and
  `cycle_aware`.
- Rooted information-ranked spanning-forest selection followed by either
  recent or information-ranked cycle factors.
- Capsule diagnostics for selected factors, independent lineages, connected
  components, cycle rank, covered-node fraction, total factor information,
  and cycle-factor information.
- A 20-seed, 200-episode controlled structural-retrieval audit with frozen
  per-seed outputs and provenance hashes.

## Compatibility

- Package version: 0.9.0.
- SQLite schema: 3.
- Current observation/query/outcome JSON input schema: 2.
- Legacy JSON input schema 1 remains accepted but cannot use v2-only fields.
- Monitor report schema: 2.
- SQLite v0/v1/v2 databases migrate in place. Version 3 adds nullable lineage
  metadata and an append-safe `derived_from_json` array without rewriting old
  observation payloads.

## Controlled result

Under an exactly matched five-factor budget, `cycle_aware` retrieval reached
96.27% node accuracy versus 16.34% for recency and 81.57% for coverage-only
selection. The paired 95% seed-bootstrap accuracy interval versus recency was
[78.77, 81.10] percentage points; the NLL difference interval was
[-1.519, -1.509]. Exact cross-family lineage echoes produced zero marginal
total-variation drift.

This benchmark is an intentionally controlled stress test. It demonstrates the
selection mechanism, not an open-domain Agent improvement claim.
