"""Matched-budget audit for lineage-aware structural retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import posterior_memory_harness

from posterior_memory_harness import (
    InMemoryMemoryStore,
    MemoryObservation,
    MemoryQuery,
    PosteriorMemoryHarness,
    s3_law,
)


POLICIES = ("recent", "coverage", "cycle_aware")
NODES = ("root", "n1", "n2", "n3")
USEFUL_EDGES = (
    ("root", "n1"),
    ("n1", "n2"),
    ("n2", "n3"),
    ("root", "n2"),
    ("n1", "n3"),
)
BUDGET = len(USEFUL_EDGES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_posterior(
    truth: int,
    size: int,
    generator: random.Random,
) -> tuple[float, ...]:
    confuser = generator.choice(
        [index for index in range(size) if index != truth]
    )
    values = np.full(size, 0.0, dtype=float)
    if generator.random() < 0.35:
        values[truth] = 0.38
        values[confuser] = 0.45
        remainder = 0.17
    else:
        values[truth] = 0.64
        values[confuser] = 0.18
        remainder = 0.18
    for index in range(size):
        if index not in {truth, confuser}:
            values[index] = remainder / (size - 2)
    return tuple(float(value) for value in values)


def make_observation(
    episode: str,
    identifier: str,
    source: str,
    target: str,
    posterior: tuple[float, ...],
    observed_at: float,
    *,
    lineage: str,
    evidence_family: str,
    derived_from: tuple[str, ...] = (),
) -> MemoryObservation:
    size = len(posterior)
    return MemoryObservation(
        observation_id=f"{episode}:{identifier}",
        namespace=episode,
        relation_type="s3-step",
        source=source,
        target=target,
        posterior=posterior,
        prior=tuple(1.0 / size for _ in range(size)),
        evidence_family=evidence_family,
        source_event_id=f"{episode}:source:{identifier}",
        lineage_root_id=f"{episode}:lineage:{lineage}",
        derived_from=derived_from,
        encoder_revision="controlled-generator:v1",
        calibration_revision="identity:v1",
        observed_at=observed_at,
    )


def build_episode(
    seed: int,
    episode_index: int,
) -> tuple[
    tuple[int, ...],
    list[MemoryObservation],
    list[MemoryObservation],
]:
    generator = random.Random(seed * 1_000_003 + episode_index)
    law = s3_law()
    truth = (law.identity,) + tuple(
        generator.randrange(law.size) for _ in NODES[1:]
    )
    node_index = {node: index for index, node in enumerate(NODES)}
    episode = f"seed-{seed}:episode-{episode_index}"
    useful = []
    for edge_index, (source, target) in enumerate(USEFUL_EDGES):
        relation = law.relative(
            truth[node_index[source]],
            truth[node_index[target]],
        )
        useful.append(
            make_observation(
                episode,
                f"useful-{edge_index}",
                source,
                target,
                local_posterior(relation, law.size, generator),
                float(edge_index + 1),
                lineage=f"useful-{edge_index}",
                evidence_family=f"sensor:{edge_index}",
            )
        )
    uniform = tuple(1.0 / law.size for _ in range(law.size))
    clutter = [
        make_observation(
            episode,
            f"clutter-{index}",
            "root",
            "n1",
            uniform,
            float(20 + index),
            lineage=f"clutter-{index}",
            evidence_family=f"weak-parser:{index}",
        )
        for index in range(BUDGET)
    ]
    return truth, useful, clutter


def score_capsule(capsule, truth: tuple[int, ...]) -> dict[str, float]:
    law = s3_law()
    labels = law.labels
    correct = 0
    nll = 0.0
    brier = 0.0
    for node_index, node in enumerate(NODES[1:], start=1):
        belief = next(
            item for item in capsule.beliefs if item.node == node
        )
        probabilities = {
            candidate.state: candidate.probability
            for candidate in belief.candidates
        }
        truth_label = labels[truth[node_index]]
        correct += belief.candidates[0].state == truth_label
        truth_probability = max(probabilities[truth_label], 1e-15)
        nll -= math.log(truth_probability)
        brier += sum(
            (
                probabilities[label]
                - float(label == truth_label)
            )
            ** 2
            for label in labels
        )
    denominator = len(NODES) - 1
    return {
        "accuracy": correct / denominator,
        "nll": nll / denominator,
        "brier": brier / denominator,
        "selected": float(
            capsule.retrieval.selected_observation_count
        ),
        "cycle_rank": float(capsule.retrieval.cycle_rank),
        "covered_node_fraction": (
            capsule.retrieval.covered_node_fraction
        ),
        "factor_information_score": (
            capsule.retrieval.factor_information_score
        ),
        "cycle_information_score": (
            capsule.retrieval.cycle_information_score
        ),
    }


def capsule_probabilities(capsule) -> dict[str, tuple[float, ...]]:
    return {
        belief.node: tuple(
            candidate.probability for candidate in belief.candidates
        )
        for belief in capsule.beliefs
    }


def max_total_variation(left, right) -> float:
    left_values = capsule_probabilities(left)
    right_values = capsule_probabilities(right)
    return max(
        0.5
        * sum(
            abs(first - second)
            for first, second in zip(
                left_values[node],
                right_values[node],
            )
        )
        for node in left_values
    )


def query_policy(
    harness: PosteriorMemoryHarness,
    namespace: str,
    policy: str,
):
    return harness.query(
        MemoryQuery(
            namespace,
            "s3-step",
            NODES,
            "root",
            1_000.0,
            max_observations=BUDGET,
            top_k=6,
            retrieval_policy=policy,
        )
    )


def run_seed(seed: int, episodes: int) -> list[dict[str, object]]:
    totals = {
        policy: {
            "accuracy": 0.0,
            "nll": 0.0,
            "brier": 0.0,
            "selected": 0.0,
            "cycle_rank": 0.0,
            "covered_node_fraction": 0.0,
            "factor_information_score": 0.0,
            "cycle_information_score": 0.0,
        }
        for policy in POLICIES
    }
    lineage_max_tv = 0.0
    for episode_index in range(episodes):
        truth, useful, clutter = build_episode(seed, episode_index)
        namespace = useful[0].namespace
        harness = PosteriorMemoryHarness(
            InMemoryMemoryStore()
        ).register_relation("s3-step", s3_law())
        for observation in useful + clutter:
            harness.observe(observation)
        capsules = {
            policy: query_policy(harness, namespace, policy)
            for policy in POLICIES
        }
        for policy, capsule in capsules.items():
            metrics = score_capsule(capsule, truth)
            for metric, value in metrics.items():
                totals[policy][metric] += value

        before = capsules["cycle_aware"]
        for observation in useful:
            harness.observe(
                make_observation(
                    namespace,
                    f"echo-{observation.observation_id}",
                    observation.source,
                    observation.target,
                    observation.posterior,
                    observation.observed_at + 100.0,
                    lineage=observation.lineage_root_id.split(
                        ":lineage:", 1
                    )[1],
                    evidence_family="model-echo",
                    derived_from=(observation.observation_id,),
                )
            )
        after = query_policy(harness, namespace, "cycle_aware")
        lineage_max_tv = max(
            lineage_max_tv,
            max_total_variation(before, after),
        )
    rows = []
    for policy in POLICIES:
        row = {
            "seed": seed,
            "policy": policy,
            **{
                metric: value / episodes
                for metric, value in totals[policy].items()
            },
            "lineage_echo_max_tv": lineage_max_tv,
        }
        rows.append(row)
    return rows


def bootstrap_interval(
    values: Iterable[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(array),
        size=(samples, len(array)),
    )
    means = np.mean(array[indices], axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize(
    rows: list[dict[str, object]],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    by_policy = {
        policy: [
            row for row in rows if row["policy"] == policy
        ]
        for policy in POLICIES
    }
    metrics = (
        "accuracy",
        "nll",
        "brier",
        "selected",
        "cycle_rank",
        "covered_node_fraction",
        "factor_information_score",
        "cycle_information_score",
    )
    absolute = {
        policy: {
            metric: float(
                np.mean([float(row[metric]) for row in values])
            )
            for metric in metrics
        }
        for policy, values in by_policy.items()
    }
    paired = {}
    cycle = {
        int(row["seed"]): row
        for row in by_policy["cycle_aware"]
    }
    for baseline in ("recent", "coverage"):
        baseline_rows = {
            int(row["seed"]): row
            for row in by_policy[baseline]
        }
        paired[baseline] = {}
        for metric_index, metric in enumerate(
            ("accuracy", "nll", "brier")
        ):
            differences = [
                float(cycle[seed][metric])
                - float(baseline_rows[seed][metric])
                for seed in sorted(cycle)
            ]
            interval = bootstrap_interval(
                differences,
                samples=bootstrap_samples,
                seed=91_000 + metric_index,
            )
            paired[baseline][metric] = {
                "mean_difference": float(np.mean(differences)),
                "ci95": list(interval),
            }
    selected_counts = {
        float(row["selected"]) for row in rows
    }
    lineage_max_tv = max(
        float(row["lineage_echo_max_tv"]) for row in rows
    )
    guards = {
        "matched_selected_count": selected_counts == {float(BUDGET)},
        "cycle_accuracy_over_recent": (
            paired["recent"]["accuracy"]["ci95"][0] > 0.0
        ),
        "cycle_nll_below_recent": (
            paired["recent"]["nll"]["ci95"][1] < 0.0
        ),
        "lineage_echo_invariant": lineage_max_tv <= 1e-12,
    }
    return {
        "schema_version": 1,
        "benchmark": "matched_budget_structural_retrieval",
        "policies": list(POLICIES),
        "nodes": list(NODES),
        "observation_budget": BUDGET,
        "absolute": absolute,
        "paired_cycle_aware_minus_baseline": paired,
        "lineage_echo_max_tv": lineage_max_tv,
        "guards": guards,
        "passed": all(guards.values()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"output directory already exists: {args.output}"
        )
    if args.seeds < 2 or args.episodes < 1:
        raise ValueError("seeds must be >=2 and episodes must be positive")
    seeds = tuple(range(7_000, 7_000 + args.seeds))
    rows = [
        row
        for seed in seeds
        for row in run_seed(seed, args.episodes)
    ]
    summary = summarize(
        rows,
        bootstrap_samples=args.bootstrap_samples,
    )
    runner = Path(__file__).resolve()
    summary["runner_sha256"] = sha256_file(runner)
    summary["package_version"] = posterior_memory_harness.__version__
    args.output.mkdir(parents=True)
    write_csv(args.output / "per_seed.csv", rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "seeds": list(seeds),
                "episodes_per_seed": args.episodes,
                "bootstrap_samples": args.bootstrap_samples,
                "observation_budget": BUDGET,
                "policies": list(POLICIES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result_files = ("per_seed.csv", "summary.json", "config.json")
    project_root = runner.parents[2]
    package_root = (
        project_root / "src" / "posterior_memory_harness"
    )
    (args.output / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "runner_sha256": sha256_file(runner),
                "package_source_sha256": {
                    path.relative_to(project_root).as_posix(): (
                        sha256_file(path)
                    )
                    for path in sorted(package_root.rglob("*.py"))
                },
                "result_sha256": {
                    name: sha256_file(args.output / name)
                    for name in result_files
                },
                "runtime": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "posterior_memory_harness": (
                        posterior_memory_harness.__version__
                    ),
                },
                "scope": (
                    "controlled matched-budget mechanism audit; "
                    "not an open-domain agent benchmark"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
