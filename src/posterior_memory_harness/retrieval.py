"""Lineage-aware, structure-preserving observation selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .models import (
    MemoryNeighborhood,
    MemoryNeighborhoodQuery,
    MemoryObservation,
)


RETRIEVAL_POLICIES = ("recent", "coverage", "cycle_aware")


@dataclass(frozen=True)
class StructuralSummary:
    selected_observation_count: int
    independent_lineage_count: int
    connected_components: int
    cycle_rank: int
    covered_node_fraction: float
    factor_information_score: float
    cycle_information_score: float


class _UnionFind:
    def __init__(self, nodes: Iterable[str]):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in self.parent}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def deduplicate_lineages(
    observations: Iterable[MemoryObservation],
) -> tuple[MemoryObservation, ...]:
    """Retain one latest independent likelihood per underlying source lineage."""
    selected: dict[str, MemoryObservation] = {}
    for observation in observations:
        if not observation.independent_evidence:
            continue
        previous = selected.get(observation.lineage_key)
        key = (observation.observed_at, observation.observation_id)
        if previous is None or key > (
            previous.observed_at,
            previous.observation_id,
        ):
            selected[observation.lineage_key] = observation
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (-item.observed_at, item.observation_id),
        )
    )


def _jensen_shannon(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    midpoint = 0.5 * (left_values + right_values)

    def divergence(source: np.ndarray) -> float:
        mask = source > 0.0
        return float(
            np.sum(source[mask] * np.log(source[mask] / midpoint[mask]))
        )

    return 0.5 * (divergence(left_values) + divergence(right_values))


def observation_information_score(observation: MemoryObservation) -> float:
    """Bounded local utility used only for deterministic budget ordering."""
    size = len(observation.posterior)
    if size <= 1:
        ambiguity = 0.0
    else:
        entropy = -sum(
            probability * math.log(probability)
            for probability in observation.posterior
            if probability > 0.0
        )
        ambiguity = entropy / math.log(size)
    prior_change = _jensen_shannon(
        observation.posterior,
        observation.prior,
    )
    # Moderate ambiguity remains valuable because a retained cycle can resolve
    # it; a posterior identical to its prior still receives zero score.
    return (
        float(observation.trust)
        * prior_change
        * (0.5 + 0.5 * ambiguity)
    )


def _information_order(
    observations: Iterable[MemoryObservation],
) -> list[MemoryObservation]:
    return sorted(
        observations,
        key=lambda item: (
            -observation_information_score(item),
            -item.observed_at,
            item.observation_id,
        ),
    )


def _rooted_maximum_forest(
    observations: tuple[MemoryObservation, ...],
    nodes: tuple[str, ...],
    root: str,
) -> tuple[
    tuple[MemoryObservation, ...],
    tuple[MemoryObservation, ...],
]:
    node_set = set(nodes)
    ordered = [
        observation
        for observation in _information_order(observations)
        if observation.source in node_set and observation.target in node_set
    ]
    union = _UnionFind(nodes)
    maximum_forest = [
        observation
        for observation in ordered
        if union.union(observation.source, observation.target)
    ]
    priority = {
        observation.observation_id: index
        for index, observation in enumerate(ordered)
    }
    adjacency: dict[str, list[MemoryObservation]] = {
        node: [] for node in nodes
    }
    for observation in maximum_forest:
        adjacency[observation.source].append(observation)
        adjacency[observation.target].append(observation)
    for values in adjacency.values():
        values.sort(
            key=lambda item: priority[item.observation_id]
        )

    forest: list[MemoryObservation] = []
    traversed_edges: set[str] = set()
    traversed_nodes: set[str] = set()
    starts = (root,) + tuple(
        node for node in nodes if node != root
    )
    for start in starts:
        if start in traversed_nodes:
            continue
        traversed_nodes.add(start)
        frontier = deque([start])
        while frontier:
            node = frontier.popleft()
            for observation in adjacency[node]:
                if observation.observation_id in traversed_edges:
                    continue
                traversed_edges.add(observation.observation_id)
                forest.append(observation)
                other = (
                    observation.target
                    if observation.source == node
                    else observation.source
                )
                if other not in traversed_nodes:
                    traversed_nodes.add(other)
                    frontier.append(other)
    forest_ids = {observation.observation_id for observation in forest}
    non_forest = tuple(
        observation
        for observation in ordered
        if observation.observation_id not in forest_ids
    )
    return tuple(forest), non_forest


def select_budgeted_observations(
    observations: Iterable[MemoryObservation],
    *,
    nodes: tuple[str, ...],
    root: str,
    limit: int,
    policy: str,
) -> tuple[MemoryObservation, ...]:
    """Choose exactly within ``limit`` without changing factor weights."""
    if policy not in RETRIEVAL_POLICIES:
        raise ValueError(f"unsupported retrieval policy {policy!r}")
    candidates = deduplicate_lineages(observations)
    if policy == "recent":
        selected = candidates[:limit]
    else:
        forest, non_forest = _rooted_maximum_forest(
            candidates,
            nodes,
            root,
        )
        if policy == "coverage":
            remainder = tuple(
                sorted(
                    non_forest,
                    key=lambda item: (
                        -item.observed_at,
                        item.observation_id,
                    ),
                )
            )
        else:
            remainder = non_forest
        selected = (forest + remainder)[:limit]
    return tuple(selected)


def structural_summary(
    nodes: tuple[str, ...],
    observations: Iterable[MemoryObservation],
    *,
    root: str,
) -> StructuralSummary:
    unique_nodes = tuple(dict.fromkeys(nodes))
    if not unique_nodes:
        return StructuralSummary(0, 0, 0, 0, 0.0, 0.0, 0.0)
    node_set = set(unique_nodes)
    selected = tuple(
        observation
        for observation in deduplicate_lineages(observations)
        if observation.source in node_set and observation.target in node_set
    )
    union = _UnionFind(unique_nodes)
    covered: set[str] = set()
    for observation in selected:
        covered.update((observation.source, observation.target))
        union.union(observation.source, observation.target)
    components = len({union.find(node) for node in unique_nodes})
    cycle_rank = max(
        0,
        len(selected) - len(unique_nodes) + components,
    )
    _, cycle_edges = _rooted_maximum_forest(
        selected,
        unique_nodes,
        root,
    )
    return StructuralSummary(
        selected_observation_count=len(selected),
        independent_lineage_count=len(
            {observation.lineage_key for observation in selected}
        ),
        connected_components=components,
        cycle_rank=cycle_rank,
        covered_node_fraction=len(covered) / len(unique_nodes),
        factor_information_score=sum(
            (
                observation_information_score(observation)
                for observation in selected
            ),
            0.0,
        ),
        cycle_information_score=sum(
            (
                observation_information_score(observation)
                for observation in cycle_edges
            ),
            0.0,
        ),
    )


def discover_structured_neighborhood(
    observations: Iterable[MemoryObservation],
    query: MemoryNeighborhoodQuery,
) -> MemoryNeighborhood:
    """Discover nodes first, then allocate a matched observation budget."""
    candidates = deduplicate_lineages(observations)
    nodes = list(query.seeds)
    node_set = set(nodes)
    frontier = set(query.seeds)
    node_limit_hit = False
    hops_explored = 0

    for hop in range(1, query.max_hops + 1):
        incident = [
            observation
            for observation in candidates
            if (
                observation.source in frontier
                or observation.target in frontier
            )
        ]
        if not incident:
            break
        incident = _information_order(incident)
        next_frontier: set[str] = set()
        for observation in incident:
            for node in (observation.source, observation.target):
                if node in node_set:
                    continue
                if len(node_set) >= query.max_nodes:
                    node_limit_hit = True
                    continue
                node_set.add(node)
                nodes.append(node)
                next_frontier.add(node)
        hops_explored = hop
        if not next_frontier:
            break
        frontier = next_frontier

    eligible = tuple(
        observation
        for observation in candidates
        if observation.source in node_set and observation.target in node_set
    )
    selected = select_budgeted_observations(
        eligible,
        nodes=tuple(nodes),
        root=query.root,
        limit=query.max_observations,
        policy=query.retrieval_policy,
    )
    return MemoryNeighborhood(
        nodes=tuple(nodes),
        observations=selected,
        hops_explored=hops_explored,
        node_limit_hit=node_limit_hit,
        observation_limit_hit=len(eligible) > len(selected),
    )
