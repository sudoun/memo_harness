"""High-level model-agnostic memory harness."""

from __future__ import annotations

from dataclasses import replace
import time
import threading
from typing import Any, Mapping
import uuid

import numpy as np

from .inference import InferenceConfig, deduplicate_observations, infer
from .laws import FiniteRelationLaw, RelationLawRegistry
from .monitoring import (
    CalibrationDriftMonitor,
    relation_law_fingerprint,
)
from .models import (
    CandidateBelief,
    MemoryCapsule,
    MemoryConflict,
    MemoryInferenceDiagnostics,
    MemoryNeighborhoodQuery,
    MemoryObservation,
    MemoryQuery,
    MemoryRetrievalInfo,
    NodeBelief,
    normalized,
)
from .store import InMemoryMemoryStore, MemoryStore
from .retrieval import structural_summary
from .validation import (
    validate_observation_payload,
    validate_query_payload,
)


class PosteriorMemoryHarness:
    """Coordinates storage, law selection, inference, and safe capsule output."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        registry: RelationLawRegistry | None = None,
        inference_config: InferenceConfig | None = None,
        monitor: CalibrationDriftMonitor | None = None,
    ):
        self.store = store or InMemoryMemoryStore()
        self.registry = registry or RelationLawRegistry()
        self.inference_config = inference_config or InferenceConfig()
        self.monitor = monitor
        self._mutation_lock = threading.RLock()
        if self.monitor is not None:
            bind_monitor = getattr(self.store, "bind_monitor", None)
            if bind_monitor is not None:
                bind_monitor(self.monitor)

    def _sync_monitor(self) -> None:
        if self.monitor is None:
            return
        sync_monitor = getattr(self.store, "sync_monitor", None)
        if sync_monitor is not None:
            sync_monitor(self.monitor)

    @staticmethod
    def _law_snapshot(
        law: FiniteRelationLaw,
    ) -> tuple[tuple[str, ...], str]:
        return law.labels, relation_law_fingerprint(law.to_dict())

    def _bind_relation_law(
        self, relation_type: str, law: FiniteRelationLaw
    ) -> None:
        bind_relation_law = getattr(self.store, "bind_relation_law", None)
        if bind_relation_law is None:
            return
        labels, fingerprint = self._law_snapshot(law)
        bind_relation_law(
            relation_type,
            labels,
            fingerprint,
            law_payload=law.to_dict(),
        )

    def _check_relation_law(
        self, relation_type: str, law: FiniteRelationLaw
    ) -> None:
        check_relation_law = getattr(self.store, "check_relation_law", None)
        if check_relation_law is None:
            return
        labels, fingerprint = self._law_snapshot(law)
        check_relation_law(
            relation_type,
            labels,
            fingerprint,
            require_registered=True,
        )

    def register_relation(
        self, relation_type: str, law: FiniteRelationLaw
    ) -> "PosteriorMemoryHarness":
        self.registry.validate_registration(relation_type, law)
        labels, fingerprint = self._law_snapshot(law)
        self._bind_relation_law(relation_type, law)
        if self.monitor is not None:
            assert_monitor_law = getattr(
                self.store, "assert_monitor_law", None
            )
            if assert_monitor_law is not None:
                assert_monitor_law(
                    relation_type,
                    law.labels,
                    fingerprint,
                    run_id=self.monitor.config.run_id,
                    law_payload=law.to_dict(),
                )
        self.registry.register(relation_type, law)
        return self

    def observe(self, observation: MemoryObservation) -> str:
        law = self.registry.get(observation.relation_type)
        if len(observation.posterior) != law.size:
            raise ValueError("observation support does not match registered law")
        with self._mutation_lock:
            # Reassert at the mutation boundary so callers cannot bypass the
            # durable registry by passing a pre-populated in-memory registry.
            self._bind_relation_law(observation.relation_type, law)
            self._sync_monitor()
            prediction = None
            if self.monitor is not None:
                prediction = self.monitor.build_prediction(
                    observation,
                    support_labels=law.labels,
                    law_fingerprint=relation_law_fingerprint(law.to_dict()),
                )
            append_monitored = getattr(
                self.store, "append_monitored", None
            )
            if prediction is not None and append_monitored is not None:
                append_monitored(observation, prediction)
                # Pull by contiguous sequence so a concurrent writer cannot
                # create a skipped high-water gap.
                self._sync_monitor()
            else:
                tracked = False
                if prediction is not None and self.monitor is not None:
                    tracked = self.monitor.restore_prediction(prediction)
                try:
                    self.store.append(observation)
                except Exception:
                    if tracked and self.monitor is not None:
                        self.monitor.discard_unlabeled(
                            observation.observation_id
                        )
                    raise
        return observation.observation_id

    def record_outcome(
        self,
        observation_id: str,
        true_relation: str,
        labeled_at: float | None = None,
        *,
        outcome_event_id: str | None = None,
        recorded_at: float | None = None,
    ) -> bool:
        if self.monitor is None:
            raise RuntimeError("no calibration monitor is configured")
        with self._mutation_lock:
            self._sync_monitor()
            relation_type = self.monitor.relation_type_for(observation_id)
            law = self.registry.get(relation_type)
            event = self.monitor.prepare_outcome(
                observation_id,
                true_relation,
                law.labels,
                labeled_at=labeled_at,
                recorded_at=recorded_at,
                outcome_event_id=outcome_event_id,
            )
            if event is None:
                return False
            append_outcome = getattr(
                self.store, "append_monitor_outcome", None
            )
            if append_outcome is not None:
                _, inserted = append_outcome(event)
                self._sync_monitor()
                return inserted
            return self.monitor.apply_outcome(event, replay=True)

    def correct_outcome(
        self,
        observation_id: str,
        true_relation: str,
        *,
        reason: str,
        labeled_at: float | None = None,
        outcome_event_id: str | None = None,
        recorded_at: float | None = None,
    ) -> bool:
        """Append an auditable correction; the previous truth is retained."""
        if self.monitor is None:
            raise RuntimeError("no calibration monitor is configured")
        with self._mutation_lock:
            self._sync_monitor()
            relation_type = self.monitor.relation_type_for(observation_id)
            law = self.registry.get(relation_type)
            event = self.monitor.prepare_outcome(
                observation_id,
                true_relation,
                law.labels,
                labeled_at=labeled_at,
                recorded_at=recorded_at,
                outcome_event_id=outcome_event_id,
                correction_reason=reason,
            )
            if event is None:
                return False
            append_outcome = getattr(
                self.store, "append_monitor_outcome", None
            )
            if append_outcome is not None:
                _, inserted = append_outcome(event)
                self._sync_monitor()
                return inserted
            return self.monitor.apply_outcome(event, replay=True)

    def calibration_report(
        self,
        namespace: str | None = None,
        *,
        as_of: float | None = None,
    ) -> dict[str, Any]:
        if self.monitor is None:
            raise RuntimeError("no calibration monitor is configured")
        with self._mutation_lock:
            self._sync_monitor()
            return self.monitor.report_all(namespace=namespace, as_of=as_of)

    def observe_payload(self, payload: Mapping[str, Any]) -> str:
        payload = validate_observation_payload(payload)
        relation_type = payload["relation_type"]
        law = self.registry.get(relation_type)
        observation_id = payload["observation_id"] or str(uuid.uuid4())
        posterior = law.encode_probabilities(payload["posterior"])
        prior_payload = payload["prior"]
        prior = (
            tuple(1.0 / law.size for _ in range(law.size))
            if prior_payload is None
            else law.encode_probabilities(prior_payload)
        )
        observed_at = (
            time.time()
            if payload["observed_at"] is None
            else payload["observed_at"]
        )
        observation = MemoryObservation(
            observation_id=observation_id,
            namespace=payload["namespace"],
            relation_type=relation_type,
            source=payload["source"],
            target=payload["target"],
            posterior=posterior,
            prior=prior,
            evidence_family=payload["evidence_family"],
            provenance=payload["provenance"],
            observed_at=observed_at,
            valid_from=payload["valid_from"],
            valid_until=payload["valid_until"],
            independent_evidence=payload["independent_evidence"],
            trust=payload["trust"],
            tags=payload["tags"],
            source_event_id=payload["source_event_id"],
            lineage_root_id=payload["lineage_root_id"],
            derived_from=payload["derived_from"],
            encoder_revision=payload["encoder_revision"],
            calibration_revision=payload["calibration_revision"],
        )
        return self.observe(observation)

    def revoke(self, observation_id: str, revoked_at: float | None = None) -> None:
        self.store.revoke(
            observation_id,
            float(time.time() if revoked_at is None else revoked_at),
        )

    @staticmethod
    def _prior_only_capsule(
        law: FiniteRelationLaw,
        query: MemoryQuery,
        retrieval: MemoryRetrievalInfo | None = None,
    ) -> MemoryCapsule:
        beliefs = []
        for node in query.nodes:
            if node == query.root:
                probabilities = np.zeros(law.size, dtype=float)
                probabilities[law.identity] = 1.0
            else:
                prior = query.node_priors.get(node)
                probabilities = np.asarray(
                    (
                        tuple(1.0 / law.size for _ in range(law.size))
                        if prior is None
                        else normalized(prior)
                    ),
                    dtype=float,
                )
            order = np.argsort(-probabilities)[: query.top_k]
            beliefs.append(
                NodeBelief(
                    node=node,
                    candidates=tuple(
                        CandidateBelief(
                            state=law.labels[int(state)],
                            probability=float(probabilities[state]),
                        )
                        for state in order
                    ),
                )
            )
        return MemoryCapsule(
            namespace=query.namespace,
            relation_type=query.relation_type,
            beliefs=tuple(beliefs),
            conflicts=(),
            used_observation_ids=(),
            inference_backend="prior-only",
            approximate=False,
            log_evidence=0.0,
            retrieval=retrieval or MemoryRetrievalInfo(),
            inference_diagnostics=MemoryInferenceDiagnostics(
                mode="prior-only",
                stable=True,
            ),
        )

    def _capsule_from_observations(
        self,
        law: FiniteRelationLaw,
        query: MemoryQuery,
        observations: list[MemoryObservation],
        retrieval: MemoryRetrievalInfo,
    ) -> MemoryCapsule:
        selected = deduplicate_observations(observations)
        structure = structural_summary(
            query.nodes,
            selected,
            root=query.root,
        )
        retrieval = replace(
            retrieval,
            policy=query.retrieval_policy,
            selected_observation_count=(
                structure.selected_observation_count
            ),
            independent_lineage_count=(
                structure.independent_lineage_count
            ),
            connected_components=structure.connected_components,
            cycle_rank=structure.cycle_rank,
            covered_node_fraction=structure.covered_node_fraction,
            factor_information_score=(
                structure.factor_information_score
            ),
            cycle_information_score=(
                structure.cycle_information_score
            ),
        )
        if not selected:
            return self._prior_only_capsule(law, query, retrieval)
        output = infer(law, query, list(selected), self.inference_config)
        beliefs = []
        for node_index, node in enumerate(query.nodes):
            order = np.argsort(-output.marginals[node_index])[: query.top_k]
            beliefs.append(
                NodeBelief(
                    node=node,
                    candidates=tuple(
                        CandidateBelief(
                            state=law.labels[int(state)],
                            probability=float(output.marginals[node_index, state]),
                        )
                        for state in order
                    ),
                )
            )
        node_index = {node: index for index, node in enumerate(query.nodes)}
        conflicts = []
        for observation in output.observations:
            inferred = law.relative(
                int(output.map_assignment[node_index[observation.source]]),
                int(output.map_assignment[node_index[observation.target]]),
            )
            local = int(np.argmax(observation.posterior))
            if inferred != local:
                conflicts.append(
                    MemoryConflict(
                        observation_id=observation.observation_id,
                        source=observation.source,
                        target=observation.target,
                        local_top1=law.labels[local],
                        inferred_relation=law.labels[inferred],
                    )
                )
        return MemoryCapsule(
            namespace=query.namespace,
            relation_type=query.relation_type,
            beliefs=tuple(beliefs),
            conflicts=tuple(conflicts),
            used_observation_ids=tuple(
                observation.observation_id for observation in output.observations
            ),
            inference_backend=output.backend,
            approximate=output.approximate,
            log_evidence=output.log_evidence,
            retrieval=retrieval,
            inference_diagnostics=output.diagnostics,
        )

    def query(self, query: MemoryQuery) -> MemoryCapsule:
        law = self.registry.get(query.relation_type)
        self._check_relation_law(query.relation_type, law)
        observations = self.store.retrieve(query)
        return self._capsule_from_observations(
            law,
            query,
            observations,
            MemoryRetrievalInfo(
                mode="explicit",
                policy=query.retrieval_policy,
            ),
        )

    def query_neighborhood(
        self, query: MemoryNeighborhoodQuery
    ) -> MemoryCapsule:
        law = self.registry.get(query.relation_type)
        self._check_relation_law(query.relation_type, law)
        neighborhood = self.store.retrieve_neighborhood(query)
        explicit = query.explicit(neighborhood.nodes)
        retrieval = MemoryRetrievalInfo(
            mode="neighborhood",
            seed_nodes=query.seeds,
            hops_explored=neighborhood.hops_explored,
            node_limit_hit=neighborhood.node_limit_hit,
            observation_limit_hit=neighborhood.observation_limit_hit,
            policy=query.retrieval_policy,
        )
        return self._capsule_from_observations(
            law,
            explicit,
            list(neighborhood.observations),
            retrieval,
        )

    def _node_priors(
        self, law: FiniteRelationLaw, payload: Mapping[str, Any]
    ) -> dict[str, tuple[float, ...]]:
        raw_node_priors = payload.get("node_priors", {})
        return {
            str(node): law.encode_probabilities(values)
            for node, values in raw_node_priors.items()
        }

    def query_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        payload = validate_query_payload(payload)
        relation_type = payload["relation_type"]
        law = self.registry.get(relation_type)
        has_nodes = "nodes" in payload
        has_seeds = "seeds" in payload
        if has_nodes == has_seeds:
            raise ValueError("query must provide exactly one of nodes or seeds")
        node_priors = self._node_priors(law, payload)
        if has_seeds:
            neighborhood_query = MemoryNeighborhoodQuery(
                namespace=payload["namespace"],
                relation_type=relation_type,
                seeds=payload["seeds"],
                root=payload["root"],
                as_of=(
                    time.time() if payload["as_of"] is None else payload["as_of"]
                ),
                tags=payload["tags"],
                max_hops=payload["max_hops"],
                max_nodes=payload["max_nodes"],
                max_observations=payload["max_observations"],
                top_k=min(payload["top_k"], law.size),
                hard_top1=payload["hard_top1"],
                inference=payload["inference"],
                node_priors=node_priors,
                retrieval_policy=payload["retrieval_policy"],
            )
            return self.query_neighborhood(neighborhood_query).to_dict()
        query = MemoryQuery(
            namespace=payload["namespace"],
            relation_type=relation_type,
            nodes=payload["nodes"],
            root=payload["root"],
            as_of=(
                time.time() if payload["as_of"] is None else payload["as_of"]
            ),
            tags=payload["tags"],
            max_observations=payload["max_observations"],
            top_k=min(payload["top_k"], law.size),
            hard_top1=payload["hard_top1"],
            inference=payload["inference"],
            node_priors=node_priors,
            retrieval_policy=payload["retrieval_policy"],
        )
        return self.query(query).to_dict()

    def before_model(self, query_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Framework-neutral hook: return data, never prompt instructions."""
        return self.query_payload(query_payload)
