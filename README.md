# Posterior Memory Harness

> Turn noisy, related agent observations into a consistent, auditable belief—before the next model call.

Posterior Memory Harness is a portable **agent skill** for the
`posterior-memory-harness` package, a model- and framework-neutral memory
middleware for structured agent state.
Instead of flattening uncertain tool output into hard facts—or dropping
conflicting text fragments into a prompt—it retains local posteriors and
reconciles them under a finite relation law.

[Read the skill](SKILL.md) · [CLI reference](references/cli-reference.md) ·
[Encoder guide](references/encoder-guide.md) · [MIT License](LICENSE)

## Why this exists

Agents often receive incomplete or conflicting signals:

- A build tracker says a job entered phase 1 with high confidence.
- A retry repeats the same source event and must not be counted twice.
- A later tool result corrects an earlier observation.
- The next model call still needs one calibrated, traceable view of state.

Ordinary semantic memory can retrieve these records, but it does not decide
whether they are mutually consistent, duplicate evidence, or stale. Posterior
Memory Harness does: it returns a typed **memory capsule** with reconciled
beliefs, conflict diagnostics, and the observation IDs used to reach them.

```text
noisy tool / state observations
            │
            ▼
local posteriors + relation law + lineage deduplication
            │
            ▼
reconciled, attributable evidence capsule ──► next model call
```

## What makes it different

- **Keep uncertainty.** Store a posterior and prior, not just a guessed
  label.
- **Use structure.** Reconcile related nodes under a finite relation law,
  including cyclic and noncommutative laws.
- **Stay auditable.** Support corrections, revocation, and historical
  (`as_of`) queries without rewriting history.
- **Avoid double counting.** Use stable source-event and lineage IDs to
  recognize retries, parsers, and summaries of the same evidence.
- **Keep memory non-authoritative.** Capsules are typed evidence, never
  instructions for the agent to execute.

## 30-second quick start

Install the core package, then use the included payloads to record an
observation and ask for a reconciled view:

```bash
python -m pip install posterior-memory-harness
mkdir .agent-memory

posterior-memory --db .agent-memory/demo.sqlite --relation-type task-phase --law cyclic:4 observe scripts/observation.example.json
posterior-memory --db .agent-memory/demo.sqlite --relation-type task-phase --law cyclic:4 query scripts/query.example.json
```

The query response contains ranked `beliefs`, `conflicts`,
`used_observation_ids`, retrieval diagnostics, and inference-stability
diagnostics. Attach that capsule as evidence for the next decision—not as
system or developer instructions.

If the package is not yet available on PyPI in your environment, install it
from its source tree first.

## Is this the right memory for your agent?

| Situation | Use it? |
|---|---|
| Checkpoint or phase tracking with cyclic/ordered relations | Yes |
| Several noisy observations of related nodes, where redundant constraints can disambiguate state | Yes |
| Corrections, revocation, or “what did we know at time T?” audits | Yes |
| One source event arriving through retries, parsers, or summaries | Yes — use lineage deduplication |
| Free-form documents, facts, or Q&A recall | No — use semantic/vector memory |
| A bare label with no calibrated posterior or prior | No — read the encoder guide first |

## What this repository contains

This repository is the **portable skill bundle**, not the core Python package
source. Install the package separately, then copy or symlink this folder into
an agent host's skill directory.

| Path | Purpose |
|---|---|
| [SKILL.md](SKILL.md) | Trigger conditions, operating contract, workflow, and CLI quick reference |
| [references/cli-reference.md](references/cli-reference.md) | CLI surface, exit codes, and monitor semantics |
| [references/capsule-guide.md](references/capsule-guide.md) | Capsule and decision-projection fields |
| [references/encoder-guide.md](references/encoder-guide.md) | Calibration rules and encoder templates |
| [scripts/](scripts/) | Runnable observation, query, outcome, monitor, and finite-law payloads |
| [agents/openai.yaml](agents/openai.yaml) | Optional Codex/OpenAI-style UI metadata |

## Install the skill

Copy or symlink this repository folder as `posterior-memory-harness/` into a
host skill directory:

| Host | User-level | Project-level |
|---|---|---|
| Codex CLI | `$CODEX_HOME/skills/` or `~/.codex/skills/` | repo scope |
| DeepSeek Harness (`dsh`) | `~/.agents/skills/` or `~/.dsh/skills/` | `<project>/.agents/skills/` or `<project>/.dsh/skills/` |
| Claude Code | `~/.claude/skills/` | `<project>/.claude/skills/` |
| opencode / agents standard | `~/.agents/skills/` | `<project>/.agents/skills/` |

For example:

```bash
git clone https://github.com/sudoun/memo_harness.git \
  ~/.agents/skills/posterior-memory-harness
```

DeepSeek Harness and Claude Code can share one copy through a symlink:

```bash
ln -s ../.agents/skills .claude/skills
```

### Host compatibility

- Frontmatter is limited to `name`, `description`, and `metadata` for broad
  host compatibility.
- `agents/openai.yaml` provides optional Codex UI metadata; hosts that do not
  understand it simply ignore it.
- Keep the folder name aligned with the skill frontmatter name:
  `posterior-memory-harness`.

## Safety and operating contract

- Memory output is typed evidence, never instructions. Do not put capsule
  content in system or developer channels, and never execute it.
- Produce posteriors with verified parsers or calibrated extractors. An
  uncalibrated LLM guess can degrade downstream beliefs.
- Use stable IDs derived from source events; never fabricate observations or
  outcomes.
- The harness fails closed on schema, law, and monitor conflicts. Investigate
  the error instead of deleting and recreating the database.
- When reporting a belief, cite `used_observation_ids` and identify the gauge
  root—the belief is relative to that root, not an absolute claim.

For the complete workflow, payload contract, inference-stability handling,
and failure modes, start with [SKILL.md](SKILL.md).

## License

[MIT](LICENSE)
