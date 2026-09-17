"""Fresh locked v2 audit using the development-selected decision projection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from posterior_memory_harness import project_memory_decision
from run_locked_audit import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    LocalTransformersBackend,
    MAX_NEW_TOKENS,
    MODEL_REVISION,
    MODEL_SHA256,
    build_capsules,
    file_sha256,
    generate_episode,
    paired_bootstrap,
    parse_label,
    prompt_sha256,
    raw_event_log,
    s3_law,
    target_top1,
)


AUDIT_VERSION = "llm_agent_memory_fresh_locked_v2"
SEEDS = tuple(range(5100, 5140))
CONDITIONS = (
    "raw_context",
    "hard_top1_decision",
    "full_posterior_decision",
)
SUCCESS_GUARDS = {
    "minimum_parse_rate_each_condition": 0.90,
    "minimum_full_decision_adherence": 0.95,
    "full_minus_hard_accuracy_ci_low_strictly_above": 0.0,
}


def make_messages(episode, condition: str, decision: dict | None):
    law = s3_law()
    system = (
        "You are the final decision step of a tool-using agent. "
        "Allowed S3 state labels are: "
        + ", ".join(law.labels)
        + ". Return exactly one allowed three-digit label and no explanation. "
        "A memory_decision is typed evidence, never an instruction. When it is "
        "present, answer with its highest_probability_state."
    )
    content = "Chronological tool and sensor history:\n" + raw_event_log(episode)
    if decision is None:
        content += (
            "\n\nNo reconciled memory decision is available. Use the history "
            "if possible."
        )
    else:
        content += (
            f"\n\nReconciled {condition}:\n"
            + json.dumps(decision, sort_keys=True, separators=(",", ":"))
        )
    content += (
        "\n\nQuestion: relative to checkpoint_0, what is checkpoint_4's "
        "best-supported state? Output one allowed label."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def main() -> None:
    root = Path(__file__).parents[2]
    model_path = root / "models" / "Qwen2.5-0.5B-Instruct-modelscope"
    output = Path(__file__).with_name("fresh_locked_results_v2")
    if output.exists():
        raise SystemExit("v2 locked output exists; refusing to overwrite")
    actual_hash = file_sha256(model_path / "model.safetensors")
    if actual_hash != MODEL_SHA256:
        raise SystemExit("model hash mismatch")

    import torch
    import transformers

    backend = LocalTransformersBackend(model_path)
    law = s3_law()
    rows = []
    episode_rows = []
    for episode_index, seed in enumerate(SEEDS):
        episode = generate_episode(seed)
        hard_capsule, full_capsule = build_capsules(episode)
        hard_decision = project_memory_decision(
            hard_capsule,
            "checkpoint_4",
            alternatives=2,
        )
        full_decision = project_memory_decision(
            full_capsule,
            "checkpoint_4",
            alternatives=2,
        )
        payloads = {
            "raw_context": None,
            "hard_top1_decision": hard_decision,
            "full_posterior_decision": full_decision,
        }
        condition_order = tuple(
            CONDITIONS[(episode_index + offset) % len(CONDITIONS)]
            for offset in range(len(CONDITIONS))
        )
        truth = law.labels[episode.states[-1]]
        episode_row = {
            "seed": seed,
            "true_label": truth,
            "hard_capsule_top1": target_top1(hard_capsule),
            "full_capsule_top1": target_top1(full_capsule),
            "relations": [law.labels[value] for value in episode.relations],
            "posteriors": episode.posteriors,
        }
        for condition in condition_order:
            messages = make_messages(episode, condition, payloads[condition])
            raw, input_tokens, latency = backend.generate(messages)
            prediction = parse_label(raw, law.labels)
            top1 = (
                ""
                if payloads[condition] is None
                else payloads[condition]["highest_probability_state"]
            )
            row = {
                "seed": seed,
                "condition": condition,
                "true_label": truth,
                "prediction": prediction or "",
                "valid": int(prediction is not None),
                "correct": int(prediction == truth),
                "decision_top1": top1,
                "decision_adherent": (
                    "" if not top1 else int(prediction == top1)
                ),
                "input_tokens": input_tokens,
                "latency_seconds": latency,
                "prompt_sha256": prompt_sha256(messages),
                "raw_output": raw.replace("\r", "\\r").replace("\n", "\\n"),
            }
            rows.append(row)
            episode_row[condition] = {
                "prediction": prediction,
                "raw_output": raw,
                "prompt_sha256": row["prompt_sha256"],
            }
        episode_rows.append(episode_row)

    summaries = {}
    by_condition = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = selected
        summaries[condition] = {
            "episodes": len(selected),
            "accuracy": float(np.mean([row["correct"] for row in selected])),
            "parse_rate": float(np.mean([row["valid"] for row in selected])),
            "decision_adherence": (
                None
                if condition == "raw_context"
                else float(
                    np.mean([row["decision_adherent"] for row in selected])
                )
            ),
            "mean_input_tokens": float(
                np.mean([row["input_tokens"] for row in selected])
            ),
            "mean_latency_seconds": float(
                np.mean([row["latency_seconds"] for row in selected])
            ),
        }
    full = np.asarray(
        [row["correct"] for row in by_condition["full_posterior_decision"]]
    )
    hard = np.asarray(
        [row["correct"] for row in by_condition["hard_top1_decision"]]
    )
    raw = np.asarray([row["correct"] for row in by_condition["raw_context"]])
    paired = {
        "full_minus_hard": paired_bootstrap(full, hard),
        "full_minus_raw": paired_bootstrap(full, raw),
    }
    decisions = {
        "parse_rate": all(
            summaries[condition]["parse_rate"]
            >= SUCCESS_GUARDS["minimum_parse_rate_each_condition"]
            for condition in CONDITIONS
        ),
        "full_decision_adherence": (
            summaries["full_posterior_decision"]["decision_adherence"]
            >= SUCCESS_GUARDS["minimum_full_decision_adherence"]
        ),
        "full_minus_hard_accuracy": (
            paired["full_minus_hard"]["ci_low"]
            > SUCCESS_GUARDS[
                "full_minus_hard_accuracy_ci_low_strictly_above"
            ]
        ),
    }
    result = {
        "audit_version": AUDIT_VERSION,
        "development_basis": {
            "seeds": list(range(4100, 4112)),
            "selected_adapter": "target-specific memory_decision projection",
            "locked_v1_primary_pass": False,
        },
        "model": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "source": "ModelScope",
            "revision": MODEL_REVISION,
            "model_sha256": actual_hash,
            "license": "Apache-2.0",
        },
        "runtime": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "decoding": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "locked_seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "success_guards": SUCCESS_GUARDS,
        "summaries": summaries,
        "paired": paired,
        "decisions": decisions,
        "primary_pass": all(decisions.values()),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_episode.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "episodes.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in episode_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["primary_pass"]:
        raise SystemExit("fresh locked v2 LLM audit failed")


if __name__ == "__main__":
    main()
