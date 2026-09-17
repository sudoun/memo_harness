# Fresh locked local-LLM Agent memory audit

This audit checks whether a real frozen language model can consume the
structured memory capsule produced by the harness after a long, distracting
event history.

## Frozen configuration

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Source: ModelScope official Qwen repository
- Repository revision: `186d8559ad54c32cf47dc3a8225f993742c507b8`
- Weight SHA-256:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- License: Apache-2.0
- Runtime: CPU float32, greedy decoding, 20 output tokens
- Episodes: seeds 3100--3139, one run per seed and condition
- Conditions: raw context, hard-top-1 memory, full-posterior memory
- Output: one of the six registered \(S_3\) labels

Every condition receives the same chronological sensor history and distractor
events. The two memory conditions additionally receive a capsule generated
from exactly those sensor posteriors. Prompt order rotates by episode, while
decoding is deterministic.

## Pre-specified primary guards

1. parse rate at least 90% in every condition;
2. full-posterior capsule adherence at least 90%;
3. paired bootstrap 95% CI for full-minus-hard answer accuracy strictly above
   zero.

The runner refuses to overwrite an existing result directory. Per-episode raw
outputs, prompt hashes, capsule top-1 values, task posteriors, aggregate
metrics, and paired intervals are retained.

Run from the project root:

```powershell
$env:PYTHONPATH = "src"
python benchmark/llm_agent_audit/run_locked_audit.py `
  --model D:/models/Qwen2.5-0.5B-Instruct-modelscope `
  --output tmp/llm-audit-v1-rerun

# The v2 runner is hash-preserved with the frozen result. In a disposable
# copy, place the checkpoint at models/Qwen2.5-0.5B-Instruct-modelscope,
# remove only that copy's bundled fresh_locked_results_v2 directory, then run:
python benchmark/llm_agent_audit/run_locked_audit_v2.py
```

Model weights are not distributed in the source ZIP. Supply the official
ModelScope checkpoint at repository revision
`186d8559ad54c32cf47dc3a8225f993742c507b8`; the runner verifies the bundled
weight SHA-256 before loading it. The frozen audit used Python CPU execution,
PyTorch `2.12.1+cpu`, and Transformers `4.53.2`. The v1 output path must be
new; both historical runners refuse to overwrite bundled results. The v2
runner is intentionally left byte-for-byte unchanged because its SHA-256 is
recorded in the frozen summary.

This is a real local language-model evaluation, but it remains a controlled
synthetic relation-memory task. It is not evidence for open-domain assistants
or proprietary frontier models.

## Audit history

The original locked v1 output is retained in `fresh_locked_results/`. It
failed: full-capsule adherence was 15.0% and full-minus-hard answer accuracy
had a 95% CI spanning zero.

A development-only projection sweep on seeds 4100--4111 is retained in
`projection_development_results/`. It selected a compact target-specific
`memory_decision` representation; this selection was implemented as the
package-level `project_memory_decision` adapter.

Fresh locked v2 then used untouched seeds 5100--5139:

- raw-context accuracy: 10.0%;
- hard-top-1 decision accuracy: 70.0%;
- full-posterior decision accuracy: 97.5%;
- full-minus-hard: +27.5 percentage points, paired bootstrap 95% CI
  [12.5, 42.5].

All v2 primary guards passed. Its immutable outputs are in
`fresh_locked_results_v2/`.
