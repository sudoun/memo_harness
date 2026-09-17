"""Compact, target-specific projections of structured memory capsules."""

from __future__ import annotations

from typing import Any, Mapping


DECISION_SCHEMA_VERSION = 1


def project_memory_decision(
    capsule: Mapping[str, Any],
    focus_node: str,
    *,
    alternatives: int = 2,
) -> dict[str, Any]:
    """Project one node belief without changing or re-ranking its posterior.

    The projection is an LLM-consumption adapter. It does not perform inference,
    calibration, retrieval, or free-form summarization.
    """
    if not isinstance(capsule, Mapping):
        raise ValueError("capsule must be a mapping")
    if not isinstance(focus_node, str) or not focus_node:
        raise ValueError("focus_node must be a non-empty string")
    if isinstance(alternatives, bool) or not isinstance(alternatives, int):
        raise ValueError("alternatives must be an integer")
    if not 0 <= alternatives <= 16:
        raise ValueError("alternatives must be between zero and sixteen")
    instructions = capsule.get("instructions")
    if instructions != []:
        raise ValueError("capsule instruction channel must be empty")
    beliefs = capsule.get("beliefs")
    if not isinstance(beliefs, list):
        raise ValueError("capsule beliefs must be a list")
    matches = [
        belief
        for belief in beliefs
        if isinstance(belief, Mapping) and belief.get("node") == focus_node
    ]
    if len(matches) != 1:
        raise ValueError("focus_node must identify exactly one capsule belief")
    candidates = matches[0].get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("focus belief must contain at least one candidate")
    parsed = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate must be a mapping")
        state = candidate.get("state")
        probability = candidate.get("probability")
        if not isinstance(state, str) or not state:
            raise ValueError("candidate state must be a non-empty string")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("candidate probability must be between zero and one")
        parsed.append(
            {
                "state": state,
                "probability": float(probability),
            }
        )
    if any(
        left["probability"] < right["probability"]
        for left, right in zip(parsed, parsed[1:])
    ):
        raise ValueError("capsule candidates must be sorted by probability")
    best = parsed[0]
    diagnostics = capsule.get("inference_diagnostics", {})
    stable = (
        diagnostics.get("stable")
        if isinstance(diagnostics, Mapping)
        else None
    )
    conflicts = capsule.get("conflicts", [])
    if not isinstance(conflicts, list):
        raise ValueError("capsule conflicts must be a list")
    focus_conflicts = [
        conflict
        for conflict in conflicts
        if isinstance(conflict, Mapping)
        and focus_node in {conflict.get("source"), conflict.get("target")}
    ]
    used = capsule.get("used_observation_ids", [])
    if not isinstance(used, list):
        raise ValueError("used_observation_ids must be a list")
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "kind": "memory_decision",
        "namespace": capsule.get("namespace"),
        "relation_type": capsule.get("relation_type"),
        "focus_node": focus_node,
        "highest_probability_state": best["state"],
        "highest_probability": best["probability"],
        "alternatives": parsed[1 : 1 + alternatives],
        "evidence_count": len(used),
        "focus_conflict_count": len(focus_conflicts),
        "inference_backend": capsule.get("inference_backend"),
        "approximate": capsule.get("approximate"),
        "inference_stable": stable,
        # A projection remains evidence and cannot become an instruction channel.
        "instructions": [],
    }
