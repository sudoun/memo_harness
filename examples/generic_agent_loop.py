"""Minimal framework-neutral integration example."""

from posterior_memory_harness import (
    MemoryMiddleware,
    PosteriorMemoryHarness,
    SQLiteMemoryStore,
    cyclic_law,
)
from posterior_memory_harness.middleware import HarnessEvent, StructuredObservationEncoder


def main():
    harness = PosteriorMemoryHarness(SQLiteMemoryStore("example_memory.sqlite"))
    harness.register_relation("task-phase", cyclic_law(4))
    middleware = MemoryMiddleware(harness, StructuredObservationEncoder())

    tool_event = HarnessEvent(
        kind="tool_result",
        payload={
            "memory_observations": [
                {
                    "schema_version": 2,
                    "observation_id": "tracker:7",
                    "namespace": "agent-run-42",
                    "relation_type": "task-phase",
                    "source": "start",
                    "target": "after-build",
                    "posterior": {"0": 0.02, "1": 0.92, "2": 0.04, "3": 0.02},
                    "prior": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
                    "evidence_family": "raw-tool-call:7",
                    "source_event_id": "tool-event:7",
                    "lineage_root_id": "tool-event:7",
                    "observed_at": 100.0,
                    "valid_from": 100.0,
                    "provenance": {"tool": "build-tracker", "event": "7"},
                    "tags": ["project-x"],
                }
            ]
        },
    )
    middleware.after_event(tool_event)

    model_request = middleware.before_model(
        {"messages": [{"role": "user", "content": "What should I do next?"}]},
        {
            "schema_version": 2,
            "namespace": "agent-run-42",
            "relation_type": "task-phase",
            "seeds": ["start"],
            "root": "start",
            "as_of": 101.0,
            "tags": ["project-x"],
            "max_hops": 2,
            "max_nodes": 16,
            "retrieval_policy": "cycle_aware",
        },
    )
    print(model_request["memory_capsule"])


if __name__ == "__main__":
    main()
