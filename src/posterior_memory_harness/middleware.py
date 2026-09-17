"""Lifecycle middleware that can wrap any agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Protocol

from .harness import PosteriorMemoryHarness
from .models import MemoryObservation
from .projection import project_memory_decision


@dataclass(frozen=True)
class HarnessEvent:
    kind: str
    payload: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict)


class ObservationEncoder(Protocol):
    """Adapter implemented by an LLM, classifier, or verified tool parser."""

    def encode(
        self, event: HarnessEvent
    ) -> Iterable[MemoryObservation | Mapping[str, Any]]: ...


class MemoryMiddleware:
    def __init__(
        self,
        harness: PosteriorMemoryHarness,
        encoder: ObservationEncoder | None = None,
        request_field: str = "memory_capsule",
    ):
        self.harness = harness
        self.encoder = encoder
        self.request_field = request_field

    def before_model(
        self,
        model_request: Mapping[str, Any],
        memory_query: Mapping[str, Any] | None,
        focus_node: str | None = None,
        alternatives: int = 2,
    ) -> dict[str, Any]:
        output = dict(model_request)
        if memory_query is not None:
            capsule = self.harness.before_model(memory_query)
            output[self.request_field] = (
                capsule
                if focus_node is None
                else project_memory_decision(
                    capsule,
                    focus_node,
                    alternatives=alternatives,
                )
            )
        return output

    def after_event(self, event: HarnessEvent) -> list[str]:
        return self._ingest(event, force_derived=False)

    def _ingest(self, event: HarnessEvent, force_derived: bool) -> list[str]:
        if self.encoder is None:
            return []
        identifiers = []
        for value in self.encoder.encode(event):
            if isinstance(value, MemoryObservation):
                if force_derived:
                    value = replace(value, independent_evidence=False)
                identifiers.append(self.harness.observe(value))
            else:
                payload = dict(value)
                if force_derived:
                    payload["independent_evidence"] = False
                identifiers.append(self.harness.observe_payload(payload))
        return identifiers

    def after_tool(
        self, payload: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> list[str]:
        return self.after_event(HarnessEvent("tool_result", payload, context or {}))

    def after_model(
        self, payload: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> list[str]:
        return self.after_event(HarnessEvent("model_output", payload, context or {}))

    def on_compaction(
        self, payload: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> list[str]:
        """Store summaries as derived evidence, regardless of encoder defaults."""
        return self._ingest(
            HarnessEvent("compaction", payload, context or {}),
            force_derived=True,
        )


class StructuredObservationEncoder:
    """Reads already-structured observations from an event field.

    This is useful for verified tools and for harnesses whose LLM extractor
    already emits the package JSON schema.
    """

    def __init__(self, field: str = "memory_observations"):
        self.field = field

    def encode(self, event: HarnessEvent):
        values = event.payload.get(self.field, ())
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{self.field} must be a list")
        return values
