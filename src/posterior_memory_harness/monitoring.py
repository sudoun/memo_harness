"""Delayed-label calibration and prediction-drift monitoring.

The in-memory monitor is also the deterministic report engine for the durable
SQLite journal. Predictions and outcomes are immutable events; corrections
append a new event that explicitly supersedes the current outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
import threading
import time
from typing import Any, Mapping
import uuid

import numpy as np

from .models import EPS, MemoryObservation


MONITOR_REPORT_SCHEMA_VERSION = 2


class MonitorOutcomeConflictError(ValueError):
    """Outcome/correction conflicts with the immutable current journal head."""


class MonitorIdempotencyConflictError(ValueError):
    """A stable event ID was reused with a different semantic payload."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_float(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("value must be a real number")
    result = float(value)
    return 0.0 if result == 0.0 else result


def relation_law_fingerprint(law_payload: Mapping[str, object]) -> str:
    """Return a stable fingerprint for a complete finite-law payload."""
    return _sha256(dict(law_payload))


def _validate_identifier(value: str, name: str, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(
            f"{name} must be at most {maximum} characters without controls"
        )
    return value


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def validate_relation_law_snapshot(
    relation_type: object,
    support_labels: object,
    law_fingerprint: object,
) -> tuple[str, tuple[str, ...], str]:
    """Validate the immutable law identity used at durable boundaries."""
    relation = _validate_identifier(
        relation_type, "relation_type", maximum=512
    )
    if (
        isinstance(support_labels, (str, bytes))
        or not isinstance(support_labels, (tuple, list))
    ):
        raise ValueError("support_labels must be a non-empty sequence")
    labels = tuple(support_labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("support_labels must be non-empty and unique")
    for label in labels:
        _validate_identifier(label, "support label", maximum=512)
    fingerprint = _validate_sha256(
        law_fingerprint, "law_fingerprint"
    )
    return relation, labels, fingerprint


@dataclass(frozen=True)
class MonitorConfig:
    """Pre-specified windows and guards; the monitor never refits a model."""

    reference_size: int = 200
    current_size: int = 100
    min_current_size: int = 30
    ece_bins: int = 10
    max_nll_increase: float = 0.15
    max_brier_increase: float = 0.05
    max_ece: float = 0.10
    max_prediction_js: float = 0.08
    max_tracked: int = 100_000
    run_id: str = "default"

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        integer_fields = {
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "min_current_size": self.min_current_size,
            "ece_bins": self.ece_bins,
            "max_tracked": self.max_tracked,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_current_size > self.current_size:
            raise ValueError("min_current_size must not exceed current_size")
        if self.max_tracked < self.reference_size + self.current_size:
            raise ValueError(
                "max_tracked must cover the reference and current windows"
            )
        for name, value in {
            "max_nll_increase": self.max_nll_increase,
            "max_brier_increase": self.max_brier_increase,
            "max_ece": self.max_ece,
            "max_prediction_js": self.max_prediction_js,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, _canonical_float(value))

    def guard_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("run_id")
        return values

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.guard_dict())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MonitorPredictionRecord:
    observation_id: str
    run_id: str
    namespace: str
    relation_type: str
    source_family: str
    posterior: tuple[float, ...]
    support_labels: tuple[str, ...]
    law_fingerprint: str
    observed_at: float
    tracked_at: float
    payload_sha256: str
    prediction_seq: int | None = None

    def __post_init__(self) -> None:
        _validate_identifier(
            self.observation_id, "observation_id", maximum=512
        )
        _validate_identifier(self.run_id, "run_id")
        _validate_identifier(self.namespace, "namespace", maximum=512)
        _validate_identifier(
            self.relation_type, "relation_type", maximum=512
        )
        _validate_identifier(self.source_family, "source_family")
        posterior = tuple(_canonical_float(value) for value in self.posterior)
        CalibrationDriftMonitor._validate_posterior(posterior)
        labels = tuple(self.support_labels)
        if (
            len(labels) != len(posterior)
            or len(set(labels)) != len(labels)
            or any(
                not isinstance(label, str) or not label
                for label in labels
            )
        ):
            raise ValueError(
                "support_labels must be unique non-empty labels matching "
                "posterior"
            )
        for label in labels:
            _validate_identifier(label, "support label", maximum=512)
        _validate_sha256(self.law_fingerprint, "law_fingerprint")
        _validate_sha256(self.payload_sha256, "payload_sha256")
        observed_at = _canonical_float(self.observed_at)
        tracked_at = _canonical_float(self.tracked_at)
        if not math.isfinite(observed_at) or not math.isfinite(tracked_at):
            raise ValueError("prediction timestamps must be finite")
        if self.prediction_seq is not None and (
            isinstance(self.prediction_seq, bool)
            or not isinstance(self.prediction_seq, int)
            or self.prediction_seq < 1
        ):
            raise ValueError("prediction_seq must be a positive integer")
        object.__setattr__(self, "posterior", posterior)
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "tracked_at", tracked_at)

    def payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "namespace": self.namespace,
            "relation_type": self.relation_type,
            "source_family": self.source_family,
            "posterior": list(self.posterior),
            "support_labels": list(self.support_labels),
            "law_fingerprint": self.law_fingerprint,
            "observed_at": self.observed_at,
            "tracked_at": self.tracked_at,
        }

    def verify_hash(self) -> None:
        if _sha256(self.payload()) != self.payload_sha256:
            raise ValueError("monitor prediction payload hash mismatch")


@dataclass(frozen=True)
class MonitorOutcomeRecord:
    outcome_event_id: str
    observation_id: str
    relation_type: str
    true_index: int
    true_label: str
    law_fingerprint: str
    labeled_at: float
    recorded_at: float
    supersedes_event_id: str | None
    correction_reason: str | None
    payload_sha256: str
    outcome_seq: int | None = None

    def __post_init__(self) -> None:
        _validate_identifier(
            self.outcome_event_id, "outcome_event_id", maximum=512
        )
        _validate_identifier(
            self.observation_id, "observation_id", maximum=512
        )
        _validate_identifier(
            self.relation_type, "relation_type", maximum=512
        )
        _validate_identifier(self.true_label, "true_label", maximum=512)
        _validate_sha256(self.law_fingerprint, "law_fingerprint")
        _validate_sha256(self.payload_sha256, "payload_sha256")
        if (
            isinstance(self.true_index, bool)
            or not isinstance(self.true_index, int)
            or self.true_index < 0
        ):
            raise ValueError("true_index must be a non-negative integer")
        labeled_at = _canonical_float(self.labeled_at)
        recorded_at = _canonical_float(self.recorded_at)
        if not math.isfinite(labeled_at) or not math.isfinite(recorded_at):
            raise ValueError("outcome timestamps must be finite")
        if self.supersedes_event_id is not None:
            _validate_identifier(
                self.supersedes_event_id,
                "supersedes_event_id",
                maximum=512,
            )
            if (
                not isinstance(self.correction_reason, str)
                or not self.correction_reason.strip()
            ):
                raise ValueError(
                    "a superseding outcome requires a correction reason"
                )
            object.__setattr__(
                self, "correction_reason", self.correction_reason.strip()
            )
        elif self.correction_reason is not None:
            raise ValueError("an initial outcome cannot have a correction reason")
        if self.outcome_seq is not None and (
            isinstance(self.outcome_seq, bool)
            or not isinstance(self.outcome_seq, int)
            or self.outcome_seq < 1
        ):
            raise ValueError("outcome_seq must be a positive integer")
        object.__setattr__(self, "labeled_at", labeled_at)
        object.__setattr__(self, "recorded_at", recorded_at)

    def payload(self) -> dict[str, object]:
        return {
            "outcome_event_id": self.outcome_event_id,
            "observation_id": self.observation_id,
            "relation_type": self.relation_type,
            "true_index": self.true_index,
            "true_label": self.true_label,
            "law_fingerprint": self.law_fingerprint,
            "labeled_at": self.labeled_at,
            "recorded_at": self.recorded_at,
            "supersedes_event_id": self.supersedes_event_id,
            "correction_reason": self.correction_reason,
        }

    def verify_hash(self) -> None:
        if _sha256(self.payload()) != self.payload_sha256:
            raise ValueError("monitor outcome payload hash mismatch")


@dataclass(frozen=True)
class _ScoredPrediction:
    posterior: tuple[float, ...]
    true_index: int


def _metrics(
    records: list[_ScoredPrediction], bins: int
) -> dict[str, float]:
    probabilities = np.asarray([record.posterior for record in records], dtype=float)
    truth = np.asarray([record.true_index for record in records], dtype=int)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predicted == truth
    true_probability = probabilities[np.arange(len(records)), truth]
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(records)), truth] = 1.0
    bin_index = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = bin_index == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(correct[selected]))
                - float(np.mean(confidence[selected]))
            )
    top_k = min(2, probabilities.shape[1])
    top_indices = np.argpartition(
        -probabilities,
        kth=top_k - 1,
        axis=1,
    )[:, :top_k]
    top2 = np.any(top_indices == truth[:, None], axis=1)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, EPS, None)),
        axis=1,
    )
    return {
        "accuracy": float(np.mean(correct)),
        "nll": float(np.mean(-np.log(np.clip(true_probability, EPS, None)))),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": float(ece),
        "top2_coverage": float(np.mean(top2)),
        "mean_confidence": float(np.mean(confidence)),
        "mean_entropy": float(np.mean(entropy)),
    }


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def divergence(source: np.ndarray) -> float:
        selected = source > 0.0
        return float(
            np.sum(source[selected] * np.log(source[selected] / midpoint[selected]))
        )

    return 0.5 * (divergence(left) + divergence(right))


class CalibrationDriftMonitor:
    """Deterministic report engine for immutable prediction/outcome events."""

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self._records: dict[str, MonitorPredictionRecord] = {}
        self._events: dict[str, MonitorOutcomeRecord] = {}
        self._histories: dict[str, list[str]] = {}
        self._relation_metadata: dict[
            str, tuple[tuple[str, ...], str]
        ] = {}
        self._prediction_cursor = 0
        self._outcome_cursor = 0
        self._lock = threading.RLock()

    @property
    def tracked_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def prediction_high_water(self) -> int:
        with self._lock:
            return self._prediction_cursor

    @property
    def outcome_high_water(self) -> int:
        with self._lock:
            return self._outcome_cursor

    def advance_journal_cursors(
        self, prediction_seq: int, outcome_seq: int
    ) -> None:
        """Advance contiguous database scan cursors after a successful replay."""
        if prediction_seq < 0 or outcome_seq < 0:
            raise ValueError("journal cursors must be non-negative")
        with self._lock:
            if (
                prediction_seq < self._prediction_cursor
                or outcome_seq < self._outcome_cursor
            ):
                raise ValueError("journal cursors cannot move backwards")
            self._prediction_cursor = int(prediction_seq)
            self._outcome_cursor = int(outcome_seq)

    @staticmethod
    def source_family_for(observation: MemoryObservation) -> str:
        for key in ("monitor_family", "encoder_family", "tool"):
            value = observation.provenance.get(key)
            if isinstance(value, str) and value:
                return _validate_identifier(value, "source_family")
        return "unspecified"

    @staticmethod
    def _validate_posterior(values: tuple[float, ...]) -> None:
        array = np.asarray(values, dtype=float)
        if (
            array.ndim != 1
            or len(array) < 1
            or not np.all(np.isfinite(array))
            or np.any(array < 0.0)
            or not np.isclose(float(array.sum()), 1.0, atol=1e-8)
        ):
            raise ValueError(
                "monitor posterior must be a finite normalized distribution"
            )

    def build_prediction(
        self,
        observation: MemoryObservation,
        *,
        support_labels: tuple[str, ...] | None = None,
        law_fingerprint: str | None = None,
        source_family: str | None = None,
        tracked_at: float | None = None,
    ) -> MonitorPredictionRecord | None:
        if not observation.independent_evidence:
            return None
        for name in (
            "observation_id",
            "namespace",
            "relation_type",
        ):
            _validate_identifier(getattr(observation, name), name, maximum=512)
        family = _validate_identifier(
            (
                self.source_family_for(observation)
                if source_family is None
                else source_family
            ),
            "source_family",
        )
        labels = (
            tuple(str(index) for index in range(len(observation.posterior)))
            if support_labels is None
            else tuple(support_labels)
        )
        if (
            len(labels) != len(observation.posterior)
            or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError(
                "support_labels must be unique non-empty labels matching posterior"
            )
        fingerprint = (
            _sha256({"labels": list(labels), "support_only": True})
            if law_fingerprint is None
            else law_fingerprint
        )
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("law_fingerprint must be a lowercase SHA-256 hex digest")
        timestamp = _canonical_float(
            time.time() if tracked_at is None else tracked_at
        )
        if not math.isfinite(timestamp):
            raise ValueError("tracked_at must be finite")
        posterior = tuple(
            _canonical_float(value) for value in observation.posterior
        )
        observed_at = _canonical_float(observation.observed_at)
        self._validate_posterior(posterior)
        values = {
            "observation_id": observation.observation_id,
            "run_id": self.config.run_id,
            "namespace": observation.namespace,
            "relation_type": observation.relation_type,
            "source_family": family,
            "posterior": list(posterior),
            "support_labels": list(labels),
            "law_fingerprint": fingerprint,
            "observed_at": observed_at,
            "tracked_at": timestamp,
        }
        return MonitorPredictionRecord(
            observation_id=observation.observation_id,
            run_id=self.config.run_id,
            namespace=observation.namespace,
            relation_type=observation.relation_type,
            source_family=family,
            posterior=posterior,
            support_labels=labels,
            law_fingerprint=fingerprint,
            observed_at=observed_at,
            tracked_at=timestamp,
            payload_sha256=_sha256(values),
        )

    def _insert_prediction(
        self, record: MonitorPredictionRecord, *, replay: bool
    ) -> bool:
        record.verify_hash()
        if record.run_id != self.config.run_id:
            raise ValueError("monitor prediction run_id does not match config")
        _validate_identifier(record.source_family, "source_family")
        self._validate_posterior(record.posterior)
        if len(record.posterior) != len(record.support_labels):
            raise ValueError("prediction support metadata is inconsistent")
        if not all(
            math.isfinite(value) for value in (record.observed_at, record.tracked_at)
        ):
            raise ValueError("prediction timestamps must be finite")
        with self._lock:
            existing = self._records.get(record.observation_id)
            if existing is not None:
                if replay and existing.payload_sha256 == record.payload_sha256:
                    return False
                raise ValueError(
                    f"observation {record.observation_id!r} is already tracked"
                )
            if len(self._records) >= self.config.max_tracked:
                raise RuntimeError(
                    "monitor capacity reached; export/reset with a new locked "
                    "reference period"
                )
            metadata = self._relation_metadata.get(record.relation_type)
            candidate = (record.support_labels, record.law_fingerprint)
            if metadata is not None and metadata != candidate:
                raise ValueError(
                    "one relation_type cannot mix support sizes, label order, "
                    "or relation laws"
                )
            self._records[record.observation_id] = record
            self._relation_metadata[record.relation_type] = candidate
            self._histories.setdefault(record.observation_id, [])
            if record.prediction_seq is not None:
                self._prediction_cursor = max(
                    self._prediction_cursor, record.prediction_seq
                )
        return True

    def track(
        self,
        observation: MemoryObservation,
        source_family: str | None = None,
        *,
        support_labels: tuple[str, ...] | None = None,
        law_fingerprint: str | None = None,
        tracked_at: float | None = None,
    ) -> bool:
        """Register one independent posterior before its truth is known."""
        record = self.build_prediction(
            observation,
            support_labels=support_labels,
            law_fingerprint=law_fingerprint,
            source_family=source_family,
            tracked_at=tracked_at,
        )
        if record is None:
            return False
        return self._insert_prediction(record, replay=False)

    def restore_prediction(self, record: MonitorPredictionRecord) -> bool:
        """Replay one verified durable prediction; identical replay is a no-op."""
        return self._insert_prediction(record, replay=True)

    def discard_unlabeled(self, observation_id: str) -> bool:
        """Rollback a just-tracked prediction if primary storage fails."""
        with self._lock:
            record = self._records.get(observation_id)
            if record is None:
                return False
            if self._histories.get(observation_id):
                raise ValueError("a labeled prediction cannot be discarded")
            del self._records[observation_id]
            self._histories.pop(observation_id, None)
            if not any(
                value.relation_type == record.relation_type
                for value in self._records.values()
            ):
                self._relation_metadata.pop(record.relation_type, None)
        return True

    def relation_type_for(self, observation_id: str) -> str:
        with self._lock:
            try:
                return self._records[observation_id].relation_type
            except KeyError as error:
                raise KeyError(
                    f"observation {observation_id!r} is not tracked"
                ) from error

    def current_outcome(
        self, observation_id: str
    ) -> MonitorOutcomeRecord | None:
        with self._lock:
            if observation_id not in self._records:
                raise KeyError(
                    f"observation {observation_id!r} is not tracked"
                )
            history = self._histories.get(observation_id, [])
            return self._events[history[-1]] if history else None

    def prepare_outcome(
        self,
        observation_id: str,
        true_label: str,
        labels: tuple[str, ...],
        *,
        labeled_at: float | None = None,
        recorded_at: float | None = None,
        outcome_event_id: str | None = None,
        correction_reason: str | None = None,
    ) -> MonitorOutcomeRecord | None:
        """Validate and construct an immutable event without applying it."""
        if not isinstance(true_label, str) or true_label not in labels:
            raise ValueError("true_label is not in the registered relation law")
        event_id = (
            str(uuid.uuid4())
            if outcome_event_id is None
            else outcome_event_id
        )
        _validate_identifier(event_id, "outcome_event_id", maximum=512)
        now = time.time()
        recorded = _canonical_float(
            now if recorded_at is None else recorded_at
        )
        if not math.isfinite(recorded):
            raise ValueError("outcome timestamps must be finite")
        with self._lock:
            try:
                prediction = self._records[observation_id]
            except KeyError as error:
                raise KeyError(
                    f"observation {observation_id!r} is not tracked"
                ) from error
            if labels != prediction.support_labels:
                raise ValueError(
                    "registered law label order does not match prediction"
                )
            history = self._histories.get(observation_id, [])
            current = self._events[history[-1]] if history else None
            is_correction = correction_reason is not None
            reason_value = None
            if is_correction:
                if not isinstance(correction_reason, str):
                    raise ValueError("correction_reason must be a string")
                reason_value = correction_reason.strip()
                if not reason_value:
                    raise ValueError("correction_reason must be non-empty")
            effective = (
                current.labeled_at
                if is_correction and labeled_at is None and current is not None
                else _canonical_float(now if labeled_at is None else labeled_at)
            )
            if not math.isfinite(effective):
                raise ValueError("outcome timestamps must be finite")
            if effective < prediction.observed_at:
                raise ValueError("labeled_at must not precede observed_at")
            true_index = labels.index(true_label)
            existing_event = self._events.get(event_id)
            if existing_event is not None:
                matches = (
                    existing_event.observation_id == observation_id
                    and existing_event.true_index == true_index
                    and existing_event.true_label == true_label
                    and existing_event.correction_reason == reason_value
                    and (
                        labeled_at is None
                        or existing_event.labeled_at == effective
                    )
                    and (
                        recorded_at is None
                        or existing_event.recorded_at == recorded
                    )
                )
                if matches:
                    return None
                raise MonitorIdempotencyConflictError(
                    f"outcome event id {event_id!r} was reused with "
                    "different semantics"
                )
            if current is not None and not is_correction:
                if current.true_index == true_index:
                    return None
                raise MonitorOutcomeConflictError(
                    "an existing outcome cannot be relabeled; append an "
                    "explicit correction"
                )
            if current is None and is_correction:
                raise MonitorOutcomeConflictError(
                    "a correction requires an existing outcome"
                )
            if current is not None and is_correction:
                if current.true_index == true_index:
                    return None
                if effective < current.labeled_at:
                    raise ValueError(
                        "a correction cannot precede the current outcome"
                    )
                if recorded < current.recorded_at:
                    raise ValueError(
                        "a correction cannot be recorded before its predecessor"
                    )
            supersedes = (
                None if current is None else current.outcome_event_id
            )
            values = {
                "outcome_event_id": event_id,
                "observation_id": observation_id,
                "relation_type": prediction.relation_type,
                "true_index": true_index,
                "true_label": true_label,
                "law_fingerprint": prediction.law_fingerprint,
                "labeled_at": effective,
                "recorded_at": recorded,
                "supersedes_event_id": supersedes,
                "correction_reason": reason_value,
            }
            return MonitorOutcomeRecord(
                **values,
                payload_sha256=_sha256(values),
            )

    def apply_outcome(
        self, event: MonitorOutcomeRecord, *, replay: bool = False
    ) -> bool:
        event.verify_hash()
        with self._lock:
            existing = self._events.get(event.outcome_event_id)
            if existing is not None:
                if replay and existing.payload_sha256 == event.payload_sha256:
                    return False
                raise MonitorIdempotencyConflictError(
                    f"outcome event {event.outcome_event_id!r} already exists"
                )
            try:
                prediction = self._records[event.observation_id]
            except KeyError as error:
                raise KeyError(
                    f"observation {event.observation_id!r} is not tracked"
                ) from error
            if (
                event.relation_type != prediction.relation_type
                or event.law_fingerprint != prediction.law_fingerprint
                or not 0 <= event.true_index < len(prediction.support_labels)
                or prediction.support_labels[event.true_index] != event.true_label
            ):
                raise ValueError("outcome does not match its prediction snapshot")
            if event.labeled_at < prediction.observed_at:
                raise ValueError("labeled_at must not precede observed_at")
            history = self._histories.setdefault(event.observation_id, [])
            current_id = history[-1] if history else None
            if event.supersedes_event_id != current_id:
                raise MonitorOutcomeConflictError(
                    "outcome event does not extend the current correction chain"
                )
            if current_id is None and event.correction_reason is not None:
                raise ValueError("initial outcome cannot be a correction")
            if current_id is not None:
                previous = self._events[current_id]
                if not event.correction_reason:
                    raise ValueError("correction_reason is required")
                if (
                    event.labeled_at < previous.labeled_at
                    or event.recorded_at < previous.recorded_at
                ):
                    raise ValueError(
                        "correction timestamps must not precede predecessor"
                    )
            self._events[event.outcome_event_id] = event
            history.append(event.outcome_event_id)
            if event.outcome_seq is not None:
                self._outcome_cursor = max(
                    self._outcome_cursor, event.outcome_seq
                )
        return True

    def restore_outcome(self, event: MonitorOutcomeRecord) -> bool:
        return self.apply_outcome(event, replay=True)

    def label(
        self,
        observation_id: str,
        true_label: str,
        labels: tuple[str, ...],
        labeled_at: float | None = None,
        *,
        outcome_event_id: str | None = None,
        recorded_at: float | None = None,
    ) -> bool:
        """Append the first delayed ground truth; existing truth is immutable."""
        event = self.prepare_outcome(
            observation_id,
            true_label,
            labels,
            labeled_at=labeled_at,
            recorded_at=recorded_at,
            outcome_event_id=outcome_event_id,
        )
        return False if event is None else self.apply_outcome(event)

    def correct(
        self,
        observation_id: str,
        true_label: str,
        labels: tuple[str, ...],
        *,
        reason: str,
        labeled_at: float | None = None,
        recorded_at: float | None = None,
        outcome_event_id: str | None = None,
    ) -> bool:
        """Append an explicit correction while retaining the old event."""
        event = self.prepare_outcome(
            observation_id,
            true_label,
            labels,
            labeled_at=labeled_at,
            recorded_at=recorded_at,
            outcome_event_id=outcome_event_id,
            correction_reason=reason,
        )
        return False if event is None else self.apply_outcome(event)

    def _effective_outcome(
        self, observation_id: str, as_of: float | None
    ) -> MonitorOutcomeRecord | None:
        history = self._histories.get(observation_id, [])
        visible = [
            self._events[event_id]
            for event_id in history
            if as_of is None
            or (
                self._events[event_id].recorded_at <= as_of
                and self._events[event_id].labeled_at <= as_of
            )
        ]
        return visible[-1] if visible else None

    def _selected(
        self,
        *,
        namespace: str | None,
        relation_type: str | None,
        source_family: str | None,
        as_of: float | None,
    ) -> list[tuple[MonitorPredictionRecord, MonitorOutcomeRecord | None]]:
        if as_of is not None and not math.isfinite(float(as_of)):
            raise ValueError("as_of must be finite")
        with self._lock:
            values = [
                (record, self._effective_outcome(record.observation_id, as_of))
                for record in self._records.values()
                if (
                    as_of is None
                    or (
                        record.observed_at <= as_of
                        and record.tracked_at <= as_of
                    )
                )
            ]
        return [
            value
            for value in values
            if (namespace is None or value[0].namespace == namespace)
            and (
                relation_type is None
                or value[0].relation_type == relation_type
            )
            and (
                source_family is None
                or value[0].source_family == source_family
            )
        ]

    def report(
        self,
        *,
        namespace: str | None = None,
        relation_type: str | None = None,
        source_family: str | None = None,
        as_of: float | None = None,
    ) -> dict[str, Any]:
        selected = self._selected(
            namespace=namespace,
            relation_type=relation_type,
            source_family=source_family,
            as_of=as_of,
        )
        labeled = sorted(
            (
                (prediction, outcome)
                for prediction, outcome in selected
                if outcome is not None
            ),
            key=lambda value: (
                value[1].labeled_at,
                value[1].outcome_seq
                if value[1].outcome_seq is not None
                else 2**63 - 1,
                value[0].observation_id,
            ),
        )
        relation_types = sorted(
            {record.relation_type for record, _ in selected}
        )
        visible_sequences = [
            outcome.outcome_seq
            for _, outcome in labeled
            if outcome is not None and outcome.outcome_seq is not None
        ]
        visible_prediction_sequences = [
            prediction.prediction_seq
            for prediction, _ in selected
            if prediction.prediction_seq is not None
        ]
        base = {
            "schema_version": MONITOR_REPORT_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "config_sha256": self.config.fingerprint,
            "scope": {
                "namespace": namespace,
                "relation_type": relation_type,
                "source_family": source_family,
                "as_of": as_of,
            },
            "snapshot": {
                "high_water_prediction_seq": (
                    max(visible_prediction_sequences)
                    if visible_prediction_sequences
                    else None
                ),
                "high_water_outcome_seq": (
                    max(visible_sequences) if visible_sequences else None
                ),
            },
            "counts": {
                "tracked": len(selected),
                "labeled": len(labeled),
                "pending": len(selected) - len(labeled),
            },
            "locked_config": self.config.guard_dict(),
        }
        if len(relation_types) > 1:
            return {
                **base,
                "status": "segmented_by_relation",
                "reason": "probability supports cannot be pooled across laws",
                "relation_types": relation_types,
            }
        reference = labeled[: self.config.reference_size]
        after_reference = labeled[self.config.reference_size :]
        current = after_reference[-self.config.current_size :]
        base["counts"].update(
            {
                "reference": len(reference),
                "current": len(current),
            }
        )
        if (
            len(reference) < self.config.reference_size
            or len(current) < self.config.min_current_size
        ):
            return {
                **base,
                "status": "insufficient_data",
                "alerts": [],
            }

        def scored(
            values: list[
                tuple[MonitorPredictionRecord, MonitorOutcomeRecord]
            ],
        ) -> list[_ScoredPrediction]:
            return [
                _ScoredPrediction(
                    posterior=prediction.posterior,
                    true_index=outcome.true_index,
                )
                for prediction, outcome in values
            ]

        reference_metrics = _metrics(
            scored(reference), self.config.ece_bins
        )
        current_metrics = _metrics(scored(current), self.config.ece_bins)
        delta = {
            key: current_metrics[key] - reference_metrics[key]
            for key in reference_metrics
        }
        reference_mean = np.mean(
            np.asarray([record.posterior for record, _ in reference]), axis=0
        )
        current_mean = np.mean(
            np.asarray([record.posterior for record, _ in current]), axis=0
        )
        prediction_js = _jensen_shannon(reference_mean, current_mean)
        drift = {
            "prediction_marginal_js": prediction_js,
            "mean_confidence_delta": delta["mean_confidence"],
            "mean_entropy_delta": delta["mean_entropy"],
        }
        alerts = []

        def alert(code: str, observed: float, threshold: float) -> None:
            alerts.append(
                {
                    "code": code,
                    "observed": float(observed),
                    "threshold": float(threshold),
                }
            )

        if delta["nll"] > self.config.max_nll_increase:
            alert(
                "nll_regression",
                delta["nll"],
                self.config.max_nll_increase,
            )
        if delta["brier"] > self.config.max_brier_increase:
            alert(
                "brier_regression",
                delta["brier"],
                self.config.max_brier_increase,
            )
        if current_metrics["ece"] > self.config.max_ece:
            alert("ece_limit", current_metrics["ece"], self.config.max_ece)
        if prediction_js > self.config.max_prediction_js:
            alert(
                "prediction_drift",
                prediction_js,
                self.config.max_prediction_js,
            )
        return {
            **base,
            "status": "alert" if alerts else "healthy",
            "reference": reference_metrics,
            "current": current_metrics,
            "delta": delta,
            "drift": drift,
            "alerts": alerts,
        }

    def report_all(
        self,
        namespace: str | None = None,
        *,
        as_of: float | None = None,
    ) -> dict[str, Any]:
        selected = self._selected(
            namespace=namespace,
            relation_type=None,
            source_family=None,
            as_of=as_of,
        )
        relation_types = sorted(
            {record.relation_type for record, _ in selected}
        )
        relations = {}
        for relation_type in relation_types:
            families = sorted(
                {
                    record.source_family
                    for record, _ in selected
                    if record.relation_type == relation_type
                }
            )
            relations[relation_type] = {
                "overall": self.report(
                    namespace=namespace,
                    relation_type=relation_type,
                    as_of=as_of,
                ),
                "by_source_family": {
                    family: self.report(
                        namespace=namespace,
                        relation_type=relation_type,
                        source_family=family,
                        as_of=as_of,
                    )
                    for family in families
                },
            }
        return {
            "schema_version": MONITOR_REPORT_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "config_sha256": self.config.fingerprint,
            "namespace": namespace,
            "as_of": as_of,
            "relations": relations,
        }
