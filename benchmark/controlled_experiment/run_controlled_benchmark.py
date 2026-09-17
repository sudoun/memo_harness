#!/usr/bin/env python3
"""Controlled test of posterior-preserving memory for an agent state graph."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from memory_harness import (
    MemoryAtom,
    PosteriorGlueMemoryHarness,
    normalize,
    s3_relation_law,
)


NODES = tuple(f"checkpoint_{index}" for index in range(5))
CHAIN_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4))
EXTRA_EDGES = ((0, 2), (1, 3), (2, 4), (0, 4))
METHODS = (
    "hard_top1",
    "full_posterior",
    "wrong_order",
    "stratified_shuffle",
    "oracle",
)


@dataclass
class Episode:
    states: np.ndarray
    posteriors: np.ndarray
    strong: np.ndarray
    relations: np.ndarray


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    output = np.exp(shifted)
    return output / output.sum()


def local_posterior(truth: int, strong: bool, rng: np.random.Generator, size: int):
    competitors = [value for value in range(size) if value != truth]
    competitor = int(rng.choice(competitors))
    logits = rng.normal(-1.15, 0.32, size=size)
    if strong:
        logits[truth] = rng.normal(2.75, 0.30)
        logits[competitor] = rng.normal(0.15, 0.40)
    else:
        logits[truth] = rng.normal(1.20, 0.55)
        logits[competitor] = rng.normal(1.32, 0.68)
    temperature = float(rng.uniform(0.85, 1.25))
    return softmax(logits / temperature)


def generate_episodes(seed: int, count: int) -> list[Episode]:
    law = s3_relation_law()
    rng = np.random.default_rng(seed)
    edges = CHAIN_EDGES + EXTRA_EDGES
    episodes = []
    for _ in range(count):
        states = np.empty(len(NODES), dtype=int)
        states[0] = law.identity
        states[1:] = rng.integers(0, law.size, size=len(NODES) - 1)
        posteriors, strong_flags, relations = [], [], []
        for edge_index, (source, target) in enumerate(edges):
            truth = law.relative(states[source], states[target])
            strong = bool(rng.random() < (0.30 if edge_index < len(CHAIN_EDGES) else 0.42))
            posteriors.append(local_posterior(truth, strong, rng, law.size))
            strong_flags.append(strong)
            relations.append(truth)
        episodes.append(
            Episode(
                states=states,
                posteriors=np.vstack(posteriors),
                strong=np.asarray(strong_flags, dtype=bool),
                relations=np.asarray(relations, dtype=int),
            )
        )
    return episodes


def stratified_shuffle(episodes: list[Episode], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    output = [episode.posteriors.copy() for episode in episodes]
    edge_count = episodes[0].posteriors.shape[0]
    for edge_index in range(edge_count):
        for strong in (False, True):
            indices = [
                index
                for index, episode in enumerate(episodes)
                if bool(episode.strong[edge_index]) == strong
            ]
            if len(indices) < 2:
                continue
            shift = int(rng.integers(1, len(indices)))
            donors = np.roll(np.asarray(indices), shift)
            for receiver, donor in zip(indices, donors):
                output[receiver][edge_index] = episodes[int(donor)].posteriors[edge_index]
    return output


def atoms_for_episode(
    episode_index: int,
    episode: Episode,
    posteriors: np.ndarray,
    edge_count: int,
    *,
    oracle: bool = False,
) -> list[MemoryAtom]:
    law = s3_relation_law()
    edges = (CHAIN_EDGES + EXTRA_EDGES)[:edge_count]
    atoms = []
    for edge_index, (source, target) in enumerate(edges):
        posterior = posteriors[edge_index]
        if oracle:
            posterior = np.full(law.size, 1e-9)
            posterior[int(episode.relations[edge_index])] = 1.0 - (law.size - 1) * 1e-9
        atoms.append(
            MemoryAtom(
                observation_id=f"episode-{episode_index}-edge-{edge_index}",
                source=NODES[source],
                target=NODES[target],
                posterior=posterior,
                prior=np.full(law.size, 1.0 / law.size),
                evidence_family=f"episode-{episode_index}-edge-{edge_index}",
                provenance={
                    "kind": "synthetic_llm_relation_observation",
                    "reliability_bin": "strong" if episode.strong[edge_index] else "ambiguous",
                },
                observed_at=edge_index,
            )
        )
    return atoms


def evaluate_result(result, truth: np.ndarray) -> dict[str, float]:
    marginals = result.marginals
    prediction = np.argmax(marginals, axis=1)
    mask = np.arange(len(truth)) != 0
    target_probabilities = marginals[np.arange(len(truth)), truth]
    one_hot = np.eye(marginals.shape[1])[truth]
    return {
        "node_accuracy": float(np.mean(prediction[mask] == truth[mask])),
        "endpoint_accuracy": float(prediction[-1] == truth[-1]),
        "episode_success": float(np.all(prediction[mask] == truth[mask])),
        "nll": float(-np.mean(np.log(np.clip(target_probabilities[mask], 1e-12, 1.0)))),
        "brier": float(np.mean(np.sum((marginals[mask] - one_hot[mask]) ** 2, axis=1))),
        "entropy": float(
            np.mean(
                -np.sum(
                    marginals[mask] * np.log(np.clip(marginals[mask], 1e-12, 1.0)),
                    axis=1,
                )
            )
        ),
    }


def evaluate_seed(seed: int, episode_count: int) -> tuple[list[dict], dict]:
    law = s3_relation_law()
    wrong_law = law.opposite("S3_wrong_order")
    episodes = generate_episodes(1_000_000 + seed, episode_count)
    shuffled = stratified_shuffle(episodes, 2_000_000 + seed)
    rows = []
    local_correct = []
    local_top2 = []
    for episode in episodes:
        top = np.argmax(episode.posteriors, axis=1)
        local_correct.extend(top == episode.relations)
        top2 = np.argpartition(episode.posteriors, -2, axis=1)[:, -2:]
        local_top2.extend(
            episode.relations[index] in top2[index] for index in range(len(top2))
        )

    for condition, edge_count in (("cycle_free", len(CHAIN_EDGES)), ("cycle_rich", 8)):
        accumulator = {method: [] for method in METHODS}
        runtimes = {method: [] for method in METHODS}
        for episode_index, episode in enumerate(episodes):
            base_atoms = atoms_for_episode(
                episode_index, episode, episode.posteriors, edge_count
            )
            shuffled_atoms = atoms_for_episode(
                episode_index, episode, shuffled[episode_index], edge_count
            )
            oracle_atoms = atoms_for_episode(
                episode_index,
                episode,
                episode.posteriors,
                edge_count,
                oracle=True,
            )
            specifications = {
                "hard_top1": (base_atoms, law, True),
                "full_posterior": (base_atoms, law, False),
                "wrong_order": (base_atoms, wrong_law, False),
                "stratified_shuffle": (shuffled_atoms, law, False),
                "oracle": (oracle_atoms, law, False),
            }
            harness = PosteriorGlueMemoryHarness(law)
            for method, (atoms, inference_law, hard_top1) in specifications.items():
                start = time.perf_counter()
                result = harness.query(
                    NODES,
                    NODES[0],
                    atoms=atoms,
                    law=inference_law,
                    hard_top1=hard_top1,
                )
                runtimes[method].append((time.perf_counter() - start) * 1000.0)
                accumulator[method].append(evaluate_result(result, episode.states))

        for method in METHODS:
            row = {"seed": seed, "condition": condition, "method": method}
            for metric in accumulator[method][0]:
                row[metric] = float(
                    np.mean([value[metric] for value in accumulator[method]])
                )
            row["runtime_ms_per_query"] = float(np.mean(runtimes[method]))
            rows.append(row)
    diagnostics = {
        "seed": seed,
        "local_top1_accuracy": float(np.mean(local_correct)),
        "local_top2_coverage": float(np.mean(local_top2)),
        "episodes": episode_count,
    }
    return rows, diagnostics


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000):
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.mean(values)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_summary(rows: list[dict]) -> list[dict]:
    comparisons = (
        ("full_posterior", "hard_top1", "full_minus_hard"),
        ("full_posterior", "stratified_shuffle", "full_minus_shuffle"),
        ("full_posterior", "wrong_order", "full_minus_wrong"),
        ("oracle", "full_posterior", "oracle_minus_full"),
    )
    metrics = ("node_accuracy", "endpoint_accuracy", "episode_success", "nll", "brier")
    lookup = {
        (int(row["seed"]), row["condition"], row["method"]): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    output = []
    for condition in ("cycle_free", "cycle_rich"):
        for new, baseline, comparison in comparisons:
            for metric in metrics:
                values = np.asarray(
                    [
                        float(lookup[(seed, condition, new)][metric])
                        - float(lookup[(seed, condition, baseline)][metric])
                        for seed in seeds
                    ]
                )
                mean, lower, upper = bootstrap_ci(
                    values,
                    seed=3_000_000
                    + sum(ord(character) for character in condition + comparison + metric),
                )
                output.append(
                    {
                        "condition": condition,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_difference": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "n_seeds": len(seeds),
                    }
                )
    return output


def absolute_summary(rows: list[dict]) -> list[dict]:
    output = []
    for condition in ("cycle_free", "cycle_rich"):
        for method in METHODS:
            subset = [
                row for row in rows if row["condition"] == condition and row["method"] == method
            ]
            for metric in (
                "node_accuracy",
                "endpoint_accuracy",
                "episode_success",
                "nll",
                "brier",
                "runtime_ms_per_query",
            ):
                values = np.asarray([float(row[metric]) for row in subset])
                mean, lower, upper = bootstrap_ci(
                    values,
                    seed=4_000_000
                    + sum(ord(character) for character in condition + method + metric),
                )
                output.append(
                    {
                        "condition": condition,
                        "method": method,
                        "metric": metric,
                        "mean": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "n_seeds": len(values),
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find(rows, **keys):
    return next(row for row in rows if all(row[key] == value for key, value in keys.items()))


def render_report(absolute: list[dict], paired: list[dict], diagnostics: list[dict]) -> str:
    rich_full = find(
        absolute, condition="cycle_rich", method="full_posterior", metric="node_accuracy"
    )
    rich_hard = find(
        absolute, condition="cycle_rich", method="hard_top1", metric="node_accuracy"
    )
    rich_delta = find(
        paired,
        condition="cycle_rich",
        comparison="full_minus_hard",
        metric="node_accuracy",
    )
    free_delta = find(
        paired,
        condition="cycle_free",
        comparison="full_minus_hard",
        metric="node_accuracy",
    )
    wrong = find(
        paired,
        condition="cycle_rich",
        comparison="full_minus_wrong",
        metric="node_accuracy",
    )
    shuffle = find(
        paired,
        condition="cycle_rich",
        comparison="full_minus_shuffle",
        metric="node_accuracy",
    )
    local_top1 = float(np.mean([row["local_top1_accuracy"] for row in diagnostics]))
    local_top2 = float(np.mean([row["local_top2_coverage"] for row in diagnostics]))
    return f"""# Agent Memory Harness controlled benchmark

## Result

This experiment isolates the memory layer.  A fixed observation generator emits
ambiguous local relation posteriors; no LLM weights or prompts differ between
methods.

- Local relation top-1 accuracy: {100*local_top1:.2f}%.
- Local true-relation top-2 coverage: {100*local_top2:.2f}%.
- Cycle-rich hard-label memory node accuracy: {100*float(rich_hard['mean']):.2f}%.
- Cycle-rich posterior memory node accuracy: {100*float(rich_full['mean']):.2f}%.
- Full posterior minus hard label: {100*float(rich_delta['mean_difference']):+.2f} pp,
  95% CI [{100*float(rich_delta['ci95_lower']):+.2f},
  {100*float(rich_delta['ci95_upper']):+.2f}].
- The same difference without redundant cycles: {100*float(free_delta['mean_difference']):+.2f} pp,
  95% CI [{100*float(free_delta['ci95_lower']):+.2f},
  {100*float(free_delta['ci95_upper']):+.2f}].
- Correct-law minus wrong-order memory: {100*float(wrong['mean_difference']):+.2f} pp.
- Correct sample correspondence minus stratified shuffle:
  {100*float(shuffle['mean_difference']):+.2f} pp.

## Interpretation

The benchmark can establish whether the generic harness improves structured
agent-state memory under controlled ambiguity.  It does not by itself establish
an open-domain LLM gain: the observation encoder is simulated so that memory
inference can be identified separately from language extraction.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--episodes-per-seed", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.outdir)
    rows, diagnostics = [], []
    for seed in range(args.seeds):
        print(f"seed {seed + 1}/{args.seeds}", flush=True)
        seed_rows, seed_diagnostics = evaluate_seed(seed, args.episodes_per_seed)
        rows.extend(seed_rows)
        diagnostics.append(seed_diagnostics)
    absolute = absolute_summary(rows)
    paired = paired_summary(rows)
    write_csv(output / "seed_metrics.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "absolute_bootstrap.csv", absolute)
    write_csv(output / "paired_bootstrap.csv", paired)
    (output / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "group": "S3",
                "nodes": list(NODES),
                "chain_edges": CHAIN_EDGES,
                "extra_edges": EXTRA_EDGES,
                "prior_correction": True,
                "evidence_family_deduplication": True,
                "test_used_for_tuning": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = render_report(absolute, paired, diagnostics)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
