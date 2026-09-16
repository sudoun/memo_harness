# Encoder Guide

The harness is model- and framework-neutral: the caller supplies the
encoder that turns events into observations. Encoder quality determines
everything downstream. The harness reconciles evidence; it cannot repair
an uncalibrated or fabricated posterior.

## What an observation must contain

| Field | Requirement |
|---|---|
| `observation_id` | Stable, derived from the source event (retries must collide detectably) |
| `namespace` | Isolation boundary per run/tenant |
| `relation_type` | Must match the registered law's relation type |
| `source`, `target` | Nodes the relation connects; one is often the checkpoint/phase |
| `posterior` | Full distribution over the law labels (keys or array); no booleans, no missing mass beyond rounding |
| `prior` | The distribution the encoder would emit with no evidence; used for prior correction |
| `evidence_family` | Mandatory provenance family, e.g. `raw-tool-call:7` |
| `source_event_id`, `lineage_root_id` | The original underlying event; identical across parsers/echoes/summaries |
| `observed_at`, `valid_from` | Event time and validity start |
| `encoder_revision`, `calibration_revision` | Which encoder/calibration produced this posterior |
| `provenance` | JSON object kept for audit (not copied into the capsule) |

Derived evidence (summaries, compaction output, model echoes) must set
`"independent_evidence": false` and list real parents in `derived_from`
(parents must already exist in the same namespace/relation type). The
middleware force-marks compaction output as derived even if the encoder
claims independence.

The query-time factor is `ℓ(r) ∝ posterior(r) / prior(r)`: passing an
inflated prior suppresses your own evidence, and passing a training prior
as "no evidence" double-counts it.

## Template 1 — verified tool parser (recommended)

A deterministic parser reads structured tool output and emits a
near-one-hot posterior with a small floor, uniform prior, and a fixed
`calibration_revision` like `tool-deterministic:v1`. This is the highest
reliability path; the repository's dogfood example
(`examples/tool_using_agent_dogfood.py`) feeds real test/compile tool
lifecycle events through this pattern.

Use when: the event has machine-checkable structure (exit codes, phase
markers, state fields).

## Template 2 — calibrated classifier

A trained/calibrated classifier emits `posterior(r)` over the law labels;
`prior` is the label prior at calibration time. Keep
`calibration_revision` tied to the calibration artifact, and track it via
delayed `outcome` events plus `health` (accuracy, NLL, Brier, ECE,
top-2 coverage). Recalibrate when the monitor flags drift.

Use when: events are noisy measurements (not deterministic) but you own
a labeled validation stream.

## Template 3 — LLM structured extractor (use with caution)

LLM-emitted probabilities are typically miscalibrated, and the frozen
local-LLM audit in this repository showed a small model could not use the
full capsule directly (15% accuracy and capsule adherence). After the
target-specific `decision` projection was frozen, the same memory reached
97.5% answer accuracy vs 70.0% for hard-top-1. Rules:

- Validate extractor output against the strict JSON contract before
  storage; never pass raw model text through.
- Prefer conservative, low-confidence posteriors over confident guesses;
  retain top-2 mass.
- Record `encoder_revision` and `calibration_revision` for every change.
- Feed small models the `decision` projection, not the full capsule.
- Validate any extractor change on locked, fresh seeds before trusting it
  — retain failed versions instead of overwriting them.

Use when: no structured parser exists and you can afford delayed
`outcome` labeling to monitor quality.

## Encoder checklist

- [ ] `observation_id` and `source_event_id` derive from the source event
- [ ] `lineage_root_id` set to the original event, identical across echoes
- [ ] `posterior` sums to 1 over the law labels; no string/boolean values
- [ ] `prior` reflects "no evidence", not the training prior
- [ ] `evidence_family` is stable and meaningful
- [ ] derived outputs marked `independent_evidence: false` with `derived_from`
- [ ] `encoder_revision` / `calibration_revision` updated on any change
- [ ] delayed `outcome` events flow in; `health` watched for drift
