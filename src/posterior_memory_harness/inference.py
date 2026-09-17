"""Exact and beam inference for finite relation-memory graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from numbers import Integral

import numpy as np

from .laws import FiniteRelationLaw
from .models import (
    EPS,
    MemoryInferenceDiagnostics,
    MemoryObservation,
    MemoryQuery,
    normalized,
)
from .retrieval import deduplicate_lineages


@dataclass(frozen=True)
class InferenceConfig:
    max_exact_assignments: int = 250_000
    beam_size: int = 1024
    point_mass_error: float = 1e-6
    beam_stability_check: bool = True
    beam_stability_multiplier: int = 2
    beam_marginal_tv_tolerance: float = 0.02
    beam_entropy_tolerance: float = 0.05

    def __post_init__(self) -> None:
        for name, value in {
            "max_exact_assignments": self.max_exact_assignments,
            "beam_size": self.beam_size,
            "beam_stability_multiplier": self.beam_stability_multiplier,
        }.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.max_exact_assignments < 1 or self.beam_size < 1:
            raise ValueError("inference budgets must be positive")
        if isinstance(self.point_mass_error, bool) or not math.isfinite(
            float(self.point_mass_error)
        ):
            raise ValueError("point_mass_error must be finite")
        if not 0.0 < self.point_mass_error < 1.0:
            raise ValueError("point_mass_error must be between zero and one")
        if not isinstance(self.beam_stability_check, bool):
            raise ValueError("beam_stability_check must be boolean")
        if self.beam_stability_multiplier < 2:
            raise ValueError("beam_stability_multiplier must be at least two")
        for name, value in {
            "beam_marginal_tv_tolerance": self.beam_marginal_tv_tolerance,
            "beam_entropy_tolerance": self.beam_entropy_tolerance,
        }.items():
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class InferenceOutput:
    marginals: np.ndarray
    map_assignment: np.ndarray
    log_evidence: float
    observations: tuple[MemoryObservation, ...]
    backend: str
    approximate: bool
    diagnostics: MemoryInferenceDiagnostics


def deduplicate_observations(
    observations: list[MemoryObservation],
) -> tuple[MemoryObservation, ...]:
    """Allow exactly one independent likelihood per source lineage."""
    return tuple(
        sorted(
            deduplicate_lineages(observations),
            key=lambda item: (item.observed_at, item.observation_id),
        )
    )


def factor_log_values(
    observation: MemoryObservation,
    hard_top1: bool,
    config: InferenceConfig,
) -> np.ndarray:
    likelihood = np.asarray(observation.likelihood(), dtype=float)
    if hard_top1:
        size = len(likelihood)
        if size == 1:
            hardened = np.ones(1, dtype=float)
        else:
            hardened = np.full(size, config.point_mass_error / (size - 1))
            hardened[int(np.argmax(observation.posterior))] = (
                1.0 - config.point_mass_error
            )
        likelihood = hardened
    return float(observation.trust) * np.log(np.clip(likelihood, EPS, None))


def _node_prior_log(query: MemoryQuery, node: str, size: int) -> np.ndarray:
    values = query.node_priors.get(node)
    if values is None:
        return np.zeros(size, dtype=float)
    if len(values) != size:
        raise ValueError(f"node prior for {node!r} has the wrong support")
    return np.log(np.clip(np.asarray(normalized(values)), EPS, None))


def exact_inference(
    law: FiniteRelationLaw,
    query: MemoryQuery,
    observations: tuple[MemoryObservation, ...],
    config: InferenceConfig,
) -> InferenceOutput:
    node_index = {node: index for index, node in enumerate(query.nodes)}
    free_nodes = [node for node in query.nodes if node != query.root]
    assignment_count = law.size ** len(free_nodes)
    if assignment_count > config.max_exact_assignments:
        raise ValueError(
            f"exact inference requires {assignment_count} assignments, "
            f"exceeding {config.max_exact_assignments}"
        )
    assignments = np.empty((assignment_count, len(query.nodes)), dtype=int)
    for row, values in enumerate(product(range(law.size), repeat=len(free_nodes))):
        assignments[row, node_index[query.root]] = law.identity
        for node, value in zip(free_nodes, values):
            assignments[row, node_index[node]] = int(value)

    log_score = np.zeros(assignment_count, dtype=float)
    for node in free_nodes:
        log_score += _node_prior_log(query, node, law.size)[
            assignments[:, node_index[node]]
        ]
    for observation in observations:
        source = assignments[:, node_index[observation.source]]
        target = assignments[:, node_index[observation.target]]
        relation = law.table[law.inverse[source], target]
        log_score += factor_log_values(observation, query.hard_top1, config)[relation]

    maximum = float(np.max(log_score))
    weights = np.exp(log_score - maximum)
    evidence = float(weights.sum())
    weights /= evidence
    marginals = np.zeros((len(query.nodes), law.size), dtype=float)
    for index in range(len(query.nodes)):
        marginals[index] = np.bincount(
            assignments[:, index], weights=weights, minlength=law.size
        )
    return InferenceOutput(
        marginals=marginals,
        map_assignment=assignments[int(np.argmax(weights))].copy(),
        log_evidence=maximum + math.log(evidence),
        observations=observations,
        backend="exact",
        approximate=False,
        diagnostics=MemoryInferenceDiagnostics(mode="exact", stable=True),
    )


def _beam_node_order(query: MemoryQuery, observations) -> list[str]:
    neighbors = {node: set() for node in query.nodes}
    for observation in observations:
        neighbors[observation.source].add(observation.target)
        neighbors[observation.target].add(observation.source)
    order = [query.root]
    seen = {query.root}
    cursor = 0
    while cursor < len(order):
        for neighbor in sorted(neighbors[order[cursor]]):
            if neighbor not in seen:
                seen.add(neighbor)
                order.append(neighbor)
        cursor += 1
    order.extend(node for node in query.nodes if node not in seen)
    return order


def beam_inference(
    law: FiniteRelationLaw,
    query: MemoryQuery,
    observations: tuple[MemoryObservation, ...],
    config: InferenceConfig,
    beam_size: int | None = None,
) -> InferenceOutput:
    effective_beam_size = config.beam_size if beam_size is None else beam_size
    if effective_beam_size < 1:
        raise ValueError("beam_size must be positive")
    order = _beam_node_order(query, observations)
    factors = [
        (observation, factor_log_values(observation, query.hard_top1, config))
        for observation in observations
    ]
    root_score = 0.0
    for observation, factor in factors:
        if observation.source == query.root and observation.target == query.root:
            root_score += float(factor[law.identity])
    beams: list[tuple[dict[str, int], float]] = [
        ({query.root: law.identity}, root_score)
    ]
    for node in order[1:]:
        candidates = []
        prior = _node_prior_log(query, node, law.size)
        for assignment, score in beams:
            for state in range(law.size):
                extended = dict(assignment)
                extended[node] = state
                value = score + float(prior[state])
                for observation, factor in factors:
                    if node not in {observation.source, observation.target}:
                        continue
                    other = (
                        observation.target
                        if observation.source == node
                        else observation.source
                    )
                    if other not in extended:
                        continue
                    source_state = extended[observation.source]
                    target_state = extended[observation.target]
                    relation = law.relative(source_state, target_state)
                    value += float(factor[relation])
                candidates.append((extended, value))
        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:effective_beam_size]

    scores = np.asarray([score for _, score in beams], dtype=float)
    maximum = float(np.max(scores))
    weights = np.exp(scores - maximum)
    evidence = float(weights.sum())
    weights /= evidence
    marginals = np.zeros((len(query.nodes), law.size), dtype=float)
    for weight, (assignment, _) in zip(weights, beams):
        for index, node in enumerate(query.nodes):
            marginals[index, assignment[node]] += float(weight)
    map_assignment = np.asarray(
        [beams[0][0][node] for node in query.nodes], dtype=int
    )
    return InferenceOutput(
        marginals=marginals,
        map_assignment=map_assignment,
        log_evidence=maximum + math.log(evidence),
        observations=observations,
        backend=f"beam-{effective_beam_size}",
        approximate=True,
        diagnostics=MemoryInferenceDiagnostics(
            mode="beam-unchecked",
            stable=None,
            base_beam_size=effective_beam_size,
        ),
    )


def _beam_stability_diagnostics(
    base: InferenceOutput,
    comparison: InferenceOutput,
    config: InferenceConfig,
) -> MemoryInferenceDiagnostics:
    map_agreement = bool(
        np.array_equal(base.map_assignment, comparison.map_assignment)
    )
    total_variation = 0.5 * np.sum(
        np.abs(base.marginals - comparison.marginals),
        axis=1,
    )
    max_marginal_tv = float(np.max(total_variation))
    base_entropy = -np.sum(
        base.marginals * np.log(np.clip(base.marginals, EPS, None)),
        axis=1,
    )
    comparison_entropy = -np.sum(
        comparison.marginals
        * np.log(np.clip(comparison.marginals, EPS, None)),
        axis=1,
    )
    max_entropy_delta = float(
        np.max(np.abs(base_entropy - comparison_entropy))
    )
    stable = (
        map_agreement
        and max_marginal_tv <= config.beam_marginal_tv_tolerance
        and max_entropy_delta <= config.beam_entropy_tolerance
    )
    return MemoryInferenceDiagnostics(
        mode="dual-width-beam",
        stable=stable,
        base_beam_size=config.beam_size,
        comparison_beam_size=(
            config.beam_size * config.beam_stability_multiplier
        ),
        map_agreement=map_agreement,
        max_marginal_tv=max_marginal_tv,
        max_entropy_delta=max_entropy_delta,
        marginal_tv_tolerance=config.beam_marginal_tv_tolerance,
        entropy_tolerance=config.beam_entropy_tolerance,
    )


def infer(
    law: FiniteRelationLaw,
    query: MemoryQuery,
    observations: list[MemoryObservation],
    config: InferenceConfig,
) -> InferenceOutput:
    selected = deduplicate_observations(observations)
    if not selected:
        raise ValueError("query has no independent observations")
    exact_size = law.size ** (len(query.nodes) - 1)
    backend = query.inference
    if backend == "auto":
        backend = "exact" if exact_size <= config.max_exact_assignments else "beam"
    if backend == "exact":
        return exact_inference(law, query, selected, config)
    base = beam_inference(
        law,
        query,
        selected,
        config,
        beam_size=config.beam_size,
    )
    if not config.beam_stability_check:
        return base
    comparison = beam_inference(
        law,
        query,
        selected,
        config,
        beam_size=config.beam_size * config.beam_stability_multiplier,
    )
    return replace(
        comparison,
        diagnostics=_beam_stability_diagnostics(base, comparison, config),
    )
