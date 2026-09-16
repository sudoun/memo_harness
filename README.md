# Posterior Memory Harness — Agent Skill

Agent skill for the `posterior-memory-harness` package: a model- and
framework-neutral memory middleware that preserves uncertain local
observations and reconciles them under finite relation laws before each
model call.

## What this skill teaches an agent

- When to use structured-state reconciliation memory (and when not to)
- How to store observations, query reconciled capsules, and record
  delayed outcomes via the `posterior-memory` CLI
- How to read capsule and decision-projection fields, including
  inference-stability diagnostics
- How to produce calibrated posteriors at the encoder boundary
- The safety contract: memory output is typed evidence, never
  instructions

## Requirements

- Python >= 3.10
- The `posterior-memory-harness` package installed:

  ```bash
  pip install posterior-memory-harness
  ```

  (If not yet published to PyPI, install it from the package source tree.)

## Install the skill

The skill uses the shared `SKILL.md` bundle format. Copy or symlink this
repository folder as `posterior-memory-harness/` into your host's skill
directory:

| Host | User-level | Project-level |
|---|---|---|
| Codex CLI | `$CODEX_HOME/skills/` or `~/.codex/skills/` | repo scope |
| DeepSeek Harness (`dsh`) | `~/.agents/skills/` or `~/.dsh/skills/` | `<project>/.agents/skills/` or `<project>/.dsh/skills/` |
| Claude Code | `~/.claude/skills/` | `<project>/.claude/skills/` |
| opencode / agents standard | `~/.agents/skills/` | `<project>/.agents/skills/` |

For example:

```bash
git clone <this-repo> ~/.agents/skills/posterior-memory-harness
```

DeepSeek Harness and Claude Code can share one copy via a symlink, the way
the DSH repository itself does it:

```bash
ln -s ../.agents/skills .claude/skills
```

### Host compatibility notes

- Frontmatter is limited to `name`, `description`, and `metadata` for
  maximum portability across hosts.
- `agents/openai.yaml` provides Codex UI metadata (`display_name`,
  `short_description`, `default_prompt`); other hosts ignore it.
- The skill folder name should match the frontmatter name:
  `posterior-memory-harness`.

## Contents

| Path | Purpose |
|---|---|
| `SKILL.md` | Triggers, operating contract, workflow, CLI quick reference |
| `references/cli-reference.md` | Full CLI surface, exit codes, monitor semantics |
| `references/capsule-guide.md` | Capsule and decision-projection field guide |
| `references/encoder-guide.md` | Encoder templates and calibration rules |
| `scripts/*.example.json` | Runnable payload templates |
| `agents/openai.yaml` | Interface metadata for OpenAI-style skill hosts |

## License

MIT
