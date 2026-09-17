"""Dogfood the middleware around real, fixed, local tool invocations.

This example intentionally accepts no command text from the model or user.
It demonstrates lifecycle wiring, not a general shell-execution facility.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from posterior_memory_harness import (
    MemoryMiddleware,
    PosteriorMemoryHarness,
    SQLiteMemoryStore,
    cyclic_law,
)
from posterior_memory_harness.middleware import StructuredObservationEncoder


ROOT = Path(__file__).resolve().parents[1]


def run_fixed_tool(name: str, arguments: list[str]) -> dict:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return {
        "name": name,
        "arguments": arguments,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-500:],
        "stderr_tail": completed.stderr[-500:],
    }


def observation(
    identifier: str,
    source: str,
    target: str,
    posterior: dict[str, float],
    observed_at: float,
) -> dict:
    return {
        "schema_version": 2,
        "observation_id": identifier,
        "namespace": "dogfood-run",
        "relation_type": "task-progress",
        "source": source,
        "target": target,
        "posterior": posterior,
        "prior": {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
        "evidence_family": f"raw-tool:{identifier}",
        "source_event_id": f"tool-event:{identifier}",
        "lineage_root_id": f"tool-event:{identifier}",
        "observed_at": observed_at,
        "provenance": {"tool_result_id": identifier},
        "tags": ["dogfood"],
    }


def query(as_of: float) -> dict:
    return {
        "schema_version": 2,
        "namespace": "dogfood-run",
        "relation_type": "task-progress",
        "nodes": ["start", "tested", "compiled"],
        "root": "start",
        "as_of": as_of,
        "tags": ["dogfood"],
        "retrieval_policy": "cycle_aware",
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        harness = PosteriorMemoryHarness(
            SQLiteMemoryStore(Path(directory) / "dogfood.sqlite")
        ).register_relation("task-progress", cyclic_law(3))
        middleware = MemoryMiddleware(harness, StructuredObservationEncoder())
        before = harness.before_model(query(0.0))

        tests = run_fixed_tool(
            "unit-tests", ["-m", "unittest", "discover", "-s", "tests"]
        )
        if tests["returncode"] != 0:
            print(json.dumps({"before": before, "tools": [tests]}, indent=2))
            return int(tests["returncode"])
        middleware.after_tool(
            {
                "memory_observations": [
                    observation(
                        "unit-tests",
                        "start",
                        "tested",
                        {"0": 0.01, "1": 0.98, "2": 0.01},
                        1.0,
                    )
                ]
            }
        )

        compile_result = run_fixed_tool(
            "compile", ["-m", "compileall", "-q", "src"]
        )
        if compile_result["returncode"] != 0:
            print(
                json.dumps(
                    {"before": before, "tools": [tests, compile_result]},
                    indent=2,
                )
            )
            return int(compile_result["returncode"])
        middleware.after_tool(
            {
                "memory_observations": [
                    observation(
                        "compile",
                        "tested",
                        "compiled",
                        {"0": 0.01, "1": 0.97, "2": 0.02},
                        2.0,
                    ),
                    observation(
                        "tracker-cumulative",
                        "start",
                        "compiled",
                        {"0": 0.01, "1": 0.02, "2": 0.97},
                        2.0,
                    ),
                ]
            }
        )

        # Even a faulty encoder cannot promote a compaction summary to a new
        # independent likelihood.
        summary = observation(
            "compaction-summary",
            "start",
            "compiled",
            {"0": 0.01, "1": 0.01, "2": 0.98},
            3.0,
        )
        summary["independent_evidence"] = True
        middleware.on_compaction({"memory_observations": [summary]})

        after = harness.before_model(query(3.0))
        report = {
            "before": before,
            "tools": [tests, compile_result],
            "after": after,
            "compaction_was_excluded": (
                "compaction-summary" not in after["used_observation_ids"]
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
