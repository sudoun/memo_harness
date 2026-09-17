"""Model-agnostic probabilistic relation memory for agent harnesses.

The module deliberately separates candidate retrieval/observation encoding from
structured inference.  Any LLM, classifier, or verified tool can emit a
``MemoryAtom``.  Query-time inference then combines the atoms with a typed
relation law while preserving uncertainty and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping

import numpy as np


EPS = 1e-12


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=float), EPS, None)
    return array / array.sum()


@dataclass(frozen=True)
class FiniteRelationLaw:
    """Finite compositional relation law used by the generic memory core."""

    name: str
    table: np.ndarray
    inverse: np.ndarray
    labels: tuple[str, ...]
    identity: int = 0

    def __post_init__(self) -> None:
        size = len(self.labels)
        table = np.asarray(self.table, dtype=int)
        inverse = np.asarray(self.inverse, dtype=int)
        if table.shape != (size, size):
            raise ValueError("composition table has the wrong shape")
        if inverse.shape != (size,):
            raise ValueError("inverse table has the wrong shape")
        if np.any(table < 0) or np.any(table >= size):
            raise ValueError("composition table contains an invalid label")
        for value in range(size):
            if table[self.identity, value] != value or table[value, self.identity] != value:
                raise ValueError("declared identity is not an identity")
            if table[value, inverse[value]] != self.identity:
                raise ValueError("invalid inverse table")
        for left in range(size):
            for middle in range(size):
                for right in range(size):
                    if table[table[left, middle], right] != table[left, table[middle, right]]:
                        raise ValueError("composition table is not associative")
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "inverse", inverse)

    @property
    def size(self) -> int:
        return len(self.labels)

    def compose(self, left: int, right: int) -> int:
        return int(self.table[int(left), int(right)])

    def relative(self, source_state: int, target_state: int) -> int:
        """Relation r satisfying source_state * r = target_state."""
        return self.compose(int(self.inverse[int(source_state)]), int(target_state))

    def opposite(self, name: str | None = None) -> "FiniteRelationLaw":
        """Return the order-reversed law, a controlled wrong-law backend."""
        return FiniteRelationLaw(
            name=name or f"{self.name}_opposite",
            table=self.table.T.copy(),
            inverse=self.inverse.copy(),
            labels=self.labels,
            identity=self.identity,
        )


@dataclass
class MemoryAtom:
    """One immutable observation likelihood with provenance."""

    observation_id: str
    source: str
    target: str
    posterior: np.ndarray
    prior: np.ndarray
    evidence_family: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    observed_at: int = 0
    independent_evidence: bool = True

    def __post_init__(self) -> None:
        posterior = normalize(self.posterior)
        prior = normalize(self.prior)
        if posterior.shape != prior.shape:
            raise ValueError("posterior and prior must have the same support")
        self.posterior = posterior
        self.prior = prior

    def likelihood(self) -> np.ndarray:
        """Prior-corrected factor q(r|x) / mu(r)."""
        return normalize(self.posterior / np.clip(self.prior, EPS, None))

    def hardened(self, error_mass: float = 1e-6) -> "MemoryAtom":
        values = np.full_like(self.posterior, error_mass / (len(self.posterior) - 1))
        values[int(np.argmax(self.posterior))] = 1.0 - error_mass
        return MemoryAtom(
            observation_id=f"{self.observation_id}:top1",
            source=self.source,
            target=self.target,
            posterior=values,
            prior=np.full_like(values, 1.0 / len(values)),
            evidence_family=self.evidence_family,
            provenance=self.provenance,
            observed_at=self.observed_at,
            independent_evidence=self.independent_evidence,
        )


@dataclass
class InferenceResult:
    nodes: tuple[str, ...]
    marginals: np.ndarray
    map_assignment: np.ndarray
    log_evidence: float
    used_observation_ids: tuple[str, ...]

    def capsule(self, law: FiniteRelationLaw, top_k: int = 3) -> dict[str, Any]:
        beliefs = []
        for node_index, node in enumerate(self.nodes):
            order = np.argsort(-self.marginals[node_index])[:top_k]
            beliefs.append(
                {
                    "node": node,
                    "candidates": [
                        {
                            "state": law.labels[int(value)],
                            "probability": float(self.marginals[node_index, value]),
                        }
                        for value in order
                    ],
                }
            )
        return {
            "beliefs": beliefs,
            "used_observation_ids": list(self.used_observation_ids),
            "instructions": [],
        }


class PosteriorGlueMemoryHarness:
    """Append-only memory store plus exact finite-relation query backend."""

    def __init__(self, law: FiniteRelationLaw):
        self.law = law
        self._atoms: list[MemoryAtom] = []
        self._ids: set[str] = set()

    def observe(self, atom: MemoryAtom) -> None:
        if len(atom.posterior) != self.law.size:
            raise ValueError("atom support does not match relation law")
        if atom.observation_id in self._ids:
            raise ValueError(f"duplicate observation id: {atom.observation_id}")
        self._ids.add(atom.observation_id)
        self._atoms.append(atom)

    def observations(self) -> tuple[MemoryAtom, ...]:
        return tuple(self._atoms)

    @staticmethod
    def _deduplicate(atoms: Iterable[MemoryAtom]) -> list[MemoryAtom]:
        """Use one likelihood per evidence family to avoid summary double counts."""
        selected: dict[str, MemoryAtom] = {}
        for atom in atoms:
            if not atom.independent_evidence:
                continue
            previous = selected.get(atom.evidence_family)
            if previous is None or atom.observed_at < previous.observed_at:
                selected[atom.evidence_family] = atom
        return list(selected.values())

    def query(
        self,
        nodes: Iterable[str],
        root: str,
        *,
        atoms: Iterable[MemoryAtom] | None = None,
        law: FiniteRelationLaw | None = None,
        hard_top1: bool = False,
    ) -> InferenceResult:
        relation_law = law or self.law
        node_tuple = tuple(nodes)
        if root not in node_tuple:
            raise ValueError("root must be present in nodes")
        node_index = {node: index for index, node in enumerate(node_tuple)}
        candidates = self._deduplicate(atoms if atoms is not None else self._atoms)
        candidates = [
            atom
            for atom in candidates
            if atom.source in node_index and atom.target in node_index
        ]
        if not candidates:
            raise ValueError("query graph has no observations")
        if hard_top1:
            candidates = [atom.hardened() for atom in candidates]

        free_nodes = [node for node in node_tuple if node != root]
        assignments = np.empty(
            (relation_law.size ** len(free_nodes), len(node_tuple)), dtype=int
        )
        for row, values in enumerate(product(range(relation_law.size), repeat=len(free_nodes))):
            assignments[row, node_index[root]] = relation_law.identity
            for node, value in zip(free_nodes, values):
                assignments[row, node_index[node]] = int(value)

        log_score = np.zeros(len(assignments), dtype=float)
        for atom in candidates:
            source_state = assignments[:, node_index[atom.source]]
            target_state = assignments[:, node_index[atom.target]]
            relation = relation_law.table[
                relation_law.inverse[source_state], target_state
            ]
            log_score += np.log(np.clip(atom.likelihood()[relation], EPS, None))

        maximum = float(np.max(log_score))
        weights = np.exp(log_score - maximum)
        evidence = float(weights.sum())
        weights /= evidence
        marginals = np.zeros((len(node_tuple), relation_law.size), dtype=float)
        for index in range(len(node_tuple)):
            marginals[index] = np.bincount(
                assignments[:, index], weights=weights, minlength=relation_law.size
            )
        map_assignment = assignments[int(np.argmax(weights))].copy()
        return InferenceResult(
            nodes=node_tuple,
            marginals=marginals,
            map_assignment=map_assignment,
            log_evidence=maximum + float(np.log(evidence)),
            used_observation_ids=tuple(atom.observation_id for atom in candidates),
        )

    def before_model(
        self,
        query: Mapping[str, Any],
        *,
        top_k: int = 3,
    ) -> dict[str, Any]:
        result = self.query(query["nodes"], query["root"])
        return result.capsule(self.law, top_k=top_k)


def s3_relation_law() -> FiniteRelationLaw:
    """Permutation group S3, used as a noncommutative agent-state backend."""
    from itertools import permutations

    identity = (0, 1, 2)
    values = [identity] + [value for value in permutations(range(3)) if value != identity]
    lookup = {value: index for index, value in enumerate(values)}

    def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(left[right[index]] for index in range(3))

    table = np.empty((6, 6), dtype=int)
    for left_index, left in enumerate(values):
        for right_index, right in enumerate(values):
            table[left_index, right_index] = lookup[compose(left, right)]
    inverse = np.empty(6, dtype=int)
    for index in range(6):
        inverse[index] = int(np.flatnonzero(table[index] == 0)[0])
    labels = tuple("".join(str(value + 1) for value in permutation) for permutation in values)
    return FiniteRelationLaw("S3", table, inverse, labels)
