"""Development-only diagnosis of how much capsule projection an LLM needs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_locked_audit import (
    LocalTransformersBackend,
    MAX_NEW_TOKENS,
    MODEL_SHA256,
    build_capsules,
    file_sha256,
    generate_episode,
    make_messages,
    parse_label,
    s3_law,
    target_top1,
)


SEEDS = tuple(range(4100, 4112))
CONDITIONS = ("full_capsule", "focused_capsule", "direct_top1_control")


def focused_capsule(capsule: dict) -> dict:
    target = next(
        belief
        for belief in capsule["beliefs"]
        if belief["node"] == "checkpoint_4"
    )
    return {
        "schema_version": 1,
        "namespace": capsule["namespace"],
        "relation_type": capsule["relation_type"],
        "focus": target,
        "inference_backend": capsule["inference_backend"],
        "approximate": capsule["approximate"],
        "instructions": [],
    }


def main() -> None:
    root = Path(__file__).parents[2]
    model_path = root / "models" / "Qwen2.5-0.5B-Instruct-modelscope"
    if file_sha256(model_path / "model.safetensors") != MODEL_SHA256:
        raise SystemExit("model hash mismatch")
    output = Path(__file__).with_name("projection_development_results")
    if output.exists():
        raise SystemExit("development output exists; refusing to overwrite")
    backend = LocalTransformersBackend(model_path)
    law = s3_law()
    rows = []
    for seed in SEEDS:
        episode = generate_episode(seed)
        _, full = build_capsules(episode)
        top1 = target_top1(full)
        payloads = {
            "full_capsule": full,
            "focused_capsule": focused_capsule(full),
            "direct_top1_control": {
                "target": "checkpoint_4",
                "highest_probability_state": top1,
                "instructions": [],
            },
        }
        for condition in CONDITIONS:
            messages = make_messages(
                episode,
                condition,
                payloads[condition],
            )
            raw, tokens, latency = backend.generate(messages)
            prediction = parse_label(raw, law.labels)
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "prediction": prediction or "",
                    "capsule_top1": top1,
                    "adherent": int(prediction == top1),
                    "correct": int(prediction == law.labels[episode.states[-1]]),
                    "input_tokens": tokens,
                    "latency_seconds": latency,
                    "raw_output": raw.replace("\r", "\\r").replace("\n", "\\n"),
                }
            )
    summary = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            "episodes": len(selected),
            "adherence": float(np.mean([row["adherent"] for row in selected])),
            "accuracy": float(np.mean([row["correct"] for row in selected])),
            "mean_input_tokens": float(
                np.mean([row["input_tokens"] for row in selected])
            ),
        }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_episode.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "kind": "development_only",
                "seeds": list(SEEDS),
                "max_new_tokens": MAX_NEW_TOKENS,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
