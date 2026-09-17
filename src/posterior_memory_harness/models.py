"""Serializable data models for probabilistic agent memory."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping


EPS = 1e-12
MAX_DERIVED_FROM = 256


def _strict_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _canonical_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def normalized(values) -> tuple[float, ...]:
    raw = tuple(float(value) for value in values)
    if not raw:
        raise ValueError("probability vector must be non-empty")
    if any(not math.isfinite(value) or value < 0.0 for value in raw):
        raise ValueError("probabilities must be finite and non-negative")
    result = tuple(max(value, EPS) for value in raw)
    total = sum(result)
    if not total > 0:
        raise ValueError("probability vector must have positive mass")
    return tuple(value / total for value in result)


@dataclass(frozen=True)
class MemoryObservation:
    observation_id: str
    namespace: str
    relation_type: str
    source: str
    target: str
    posterior: tuple[float, ...]
    prior: tuple[float, ...]
    evidence_family: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    observed_at: float = 0.0
    valid_from: float | None = None
    valid_until: float | None = None
    independent_evidence: bool = True
    trust: float = 1.0
    tags: tuple[str, ...] = ()
    revoked_at: float | None = None
    source_event_id: str | None = None
    lineage_root_id: str | None = None
    derived_from: tuple[str, ...] = ()
    encoder_revision: str | None = None
    calibration_revision: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "namespace",
            "relation_type",
            "source",
            "target",
            "evidence_family",
        ):
            _strict_text(getattr(self, name), name)
        for name in (
            "source_event_id",
            "lineage_root_id",
            "encoder_revision",
            "calibration_revision",
        ):
            value = getattr(self, name)
            if value is not None:
                _strict_text(value, name)
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        if type(self.independent_evidence) is not bool:
            raise ValueError("independent_evidence must be a boolean")
        if any(not isinstance(tag, str) or not tag for tag in self.tags):
            raise ValueError("tags must contain non-empty strings")
        if any(
            not isinstance(identifier, str) or not identifier
            for identifier in self.derived_from
        ):
            raise ValueError(
                "derived_from must contain non-empty observation identifiers"
            )
        if len(set(self.derived_from)) != len(self.derived_from):
            raise ValueError("derived_from must not contain duplicates")
        if len(self.derived_from) > MAX_DERIVED_FROM:
            raise ValueError(
                f"derived_from may contain at most {MAX_DERIVED_FROM} items"
            )
        if self.observation_id in self.derived_from:
            raise ValueError("an observation cannot derive from itself")
        posterior = normalized(self.posterior)
        prior = normalized(self.prior)
        if len(posterior) != len(prior):
            raise ValueError("posterior and prior supports differ")
        trust = _canonical_number(self.trust, "trust")
        if not 0.0 <= trust <= 1.0:
            raise ValueError("trust must be between zero and one")
        temporal_values = {
            "observed_at": self.observed_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "revoked_at": self.revoked_at,
        }
        normalized_times = {}
        for name, value in temporal_values.items():
            normalized_times[name] = (
                None if value is None else _canonical_number(value, name)
            )
        if (
            normalized_times["valid_from"] is not None
            and normalized_times["valid_until"] is not None
        ):
            if (
                normalized_times["valid_from"]
                > normalized_times["valid_until"]
            ):
                raise ValueError("valid_from must not exceed valid_until")
        if (
            normalized_times["revoked_at"] is not None
            and normalized_times["revoked_at"]
            < normalized_times["observed_at"]
        ):
            raise ValueError("revoked_at must not precede observed_at")
        object.__setattr__(self, "posterior", posterior)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "trust", trust)
        for name, value in normalized_times.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))
        object.__setattr__(self, "derived_from", tuple(self.derived_from))

    @property
    def lineage_key(self) -> str:
        """Stable identity used to prevent one source event entering twice."""
        return (
            self.lineage_root_id
            or self.source_event_id
            or self.evidence_family
        )

    def likelihood(self) -> tuple[float, ...]:
        """Return normalized q(r|x)/mu(r), avoiding local-prior double counts."""
        return normalized(
            posterior / max(prior, EPS)
            for posterior, prior in zip(self.posterior, self.prior)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "namespace": self.namespace,
            "relation_type": self.relation_type,
            "source": self.source,
            "target": self.target,
            "posterior": list(self.posterior),
            "prior": list(self.prior),
            "evidence_family": self.evidence_family,
            "provenance": dict(self.provenance),
            "observed_at": self.observed_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "independent_evidence": self.independent_evidence,
            "trust": self.trust,
            "tags": list(self.tags),
            "revoked_at": self.revoked_at,
            "source_event_id": self.source_event_id,
            "lineage_root_id": self.lineage_root_id,
            "derived_from": list(self.derived_from),
            "encoder_revision": self.encoder_revision,
            "calibration_revision": self.calibration_revision,
        }


@dataclass(frozen=True)
class MemoryQuery:
    namespace: str
    relation_type: str
    nodes: tuple[str, ...]
    root: str
    as_of: float
    tags: tuple[str, ...] = ()
    max_observations: int = 512
    top_k: int = 3
    hard_top1: bool = False
    inference: str = "auto"
    node_priors: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    retrieval_policy: str = "recent"

    def __post_init__(self) -> None:
        if not self.namespace or not self.relation_type:
            raise ValueError("namespace and relation_type are required")
        if self.root not in self.nodes:
            raise ValueError("root must be included in nodes")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("query nodes must be unique")
        if self.max_observations < 1 or self.top_k < 1:
            raise ValueError("query budgets must be positive")
        if self.inference not in {"auto", "exact", "beam"}:
            raise ValueError("inference must be auto, exact, or beam")
        if self.retrieval_policy not in {
            "recent",
            "coverage",
            "cycle_aware",
        }:
            raise ValueError(
                "retrieval_policy must be recent, coverage, or cycle_aware"
            )
        if not math.isfinite(float(self.as_of)):
            raise ValueError("as_of must be finite")
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))


@dataclass(frozen=True)
class MemoryNeighborhoodQuery:
    namespace: str
    relation_type: str
    seeds: tuple[str, ...]
    root: str
    as_of: float
    tags: tuple[str, ...] = ()
    max_hops: int = 2
    max_nodes: int = 32
    max_observations: int = 512
    top_k: int = 3
    hard_top1: bool = False
    inference: str = "auto"
    node_priors: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    retrieval_policy: str = "recent"

    def __post_init__(self) -> None:
        if not self.namespace or not self.relation_type:
            raise ValueError("namespace and relation_type are required")
        if not self.seeds or any(not seed for seed in self.seeds):
            raise ValueError("at least one non-empty seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("neighborhood seeds must be unique")
        if self.root not in self.seeds:
            raise ValueError("root must be included in neighborhood seeds")
        if self.max_hops < 1:
            raise ValueError("max_hops must be positive")
        if self.max_nodes < len(self.seeds):
            raise ValueError("max_nodes must cover every seed")
        if self.max_observations < 1 or self.top_k < 1:
            raise ValueError("query budgets must be positive")
        if self.inference not in {"auto", "exact", "beam"}:
            raise ValueError("inference must be auto, exact, or beam")
        if self.retrieval_policy not in {
            "recent",
            "coverage",
            "cycle_aware",
        }:
            raise ValueError(
                "retrieval_policy must be recent, coverage, or cycle_aware"
            )
        if not math.isfinite(float(self.as_of)):
            raise ValueError("as_of must be finite")
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))

    def explicit(self, nodes: tuple[str, ...]) -> MemoryQuery:
        return MemoryQuery(
            namespace=self.namespace,
            relation_type=self.relation_type,
            nodes=nodes,
            root=self.root,
            as_of=self.as_of,
            tags=self.tags,
            max_observations=self.max_observations,
            top_k=self.top_k,
            hard_top1=self.hard_top1,
            inference=self.inference,
            node_priors=self.node_priors,
            retrieval_policy=self.retrieval_policy,
        )


@dataclass(frozen=True)
class MemoryNeighborhood:
    nodes: tuple[str, ...]
    observations: tuple[MemoryObservation, ...]
    hops_explored: int
    node_limit_hit: bool
    observation_limit_hit: bool


@dataclass(frozen=True)
class CandidateBelief:
    state: str
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "probability": self.probability}


@dataclass(frozen=True)
class NodeBelief:
    node: str
    candidates: tuple[CandidateBelief, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class MemoryConflict:
    observation_id: str
    source: str
    target: str
    local_top1: str
    inferred_relation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source": self.source,
            "target": self.target,
            "local_top1": self.local_top1,
            "inferred_relation": self.inferred_relation,
        }


@dataclass(frozen=True)
class MemoryRetrievalInfo:
    mode: str = "explicit"
    seed_nodes: tuple[str, ...] = ()
    hops_explored: int = 0
    node_limit_hit: bool = False
    observation_limit_hit: bool = False
    policy: str = "recent"
    selected_observation_count: int = 0
    independent_lineage_count: int = 0
    connected_components: int = 0
    cycle_rank: int = 0
    covered_node_fraction: float = 0.0
    factor_information_score: float = 0.0
    cycle_information_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed_nodes": list(self.seed_nodes),
            "hops_explored": self.hops_explored,
            "node_limit_hit": self.node_limit_hit,
            "observation_limit_hit": self.observation_limit_hit,
            "truncated": self.node_limit_hit or self.observation_limit_hit,
            "policy": self.policy,
            "selected_observation_count": self.selected_observation_count,
            "independent_lineage_count": self.independent_lineage_count,
            "connected_components": self.connected_components,
            "cycle_rank": self.cycle_rank,
            "covered_node_fraction": self.covered_node_fraction,
            "factor_information_score": self.factor_information_score,
            "cycle_information_score": self.cycle_information_score,
        }


@dataclass(frozen=True)
class MemoryInferenceDiagnostics:
    mode: str
    stable: bool | None
    base_beam_size: int | None = None
    comparison_beam_size: int | None = None
    map_agreement: bool | None = None
    max_marginal_tv: float | None = None
    max_entropy_delta: float | None = None
    marginal_tv_tolerance: float | None = None
    entropy_tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "stable": self.stable,
            "base_beam_size": self.base_beam_size,
            "comparison_beam_size": self.comparison_beam_size,
            "map_agreement": self.map_agreement,
            "max_marginal_tv": self.max_marginal_tv,
            "max_entropy_delta": self.max_entropy_delta,
            "marginal_tv_tolerance": self.marginal_tv_tolerance,
            "entropy_tolerance": self.entropy_tolerance,
        }


@dataclass(frozen=True)
class MemoryCapsule:
    namespace: str
    relation_type: str
    beliefs: tuple[NodeBelief, ...]
    conflicts: tuple[MemoryConflict, ...]
    used_observation_ids: tuple[str, ...]
    inference_backend: str
    approximate: bool
    log_evidence: float
    retrieval: MemoryRetrievalInfo = field(default_factory=MemoryRetrievalInfo)
    inference_diagnostics: MemoryInferenceDiagnostics = field(
        default_factory=lambda: MemoryInferenceDiagnostics(
            mode="unavailable",
            stable=None,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "relation_type": self.relation_type,
            "beliefs": [belief.to_dict() for belief in self.beliefs],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "used_observation_ids": list(self.used_observation_ids),
            "inference_backend": self.inference_backend,
            "approximate": self.approximate,
            "log_evidence": self.log_evidence,
            "retrieval": self.retrieval.to_dict(),
            "inference_diagnostics": self.inference_diagnostics.to_dict(),
            # Memory content is evidence, never a new instruction channel.
            "instructions": [],
        }
