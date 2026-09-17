"""Fresh fixed 10-seed mechanism audit for the calibration/drift monitor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from posterior_memory_harness import (
    CalibrationDriftMonitor,
    MemoryObservation,
    MonitorConfig,
)


SEEDS = tuple(range(2100, 2110))
REFERENCE_SIZE = 400
CURRENT_SIZE = 400
CONFIG = MonitorConfig(
    reference_size=REFERENCE_SIZE,
    current_size=CURRENT_SIZE,
    min_current_size=CURRENT_SIZE,
    ece_bins=10,
    max_nll_increase=0.10,
    max_brier_increase=0.04,
    max_ece=0.08,
    max_prediction_js=0.01,
    max_tracked=1000,
)
SCENARIOS = ("stable", "overconfident", "prediction_shift")


def add_window(
    monitor: CalibrationDriftMonitor,
    rng: np.random.Generator,
    scenario: str,
    start: int,
    size: int,
) -> None:
    for index in range(start, start + size):
        base_confidence = float(rng.uniform(0.60, 0.90))
        if scenario == "prediction_shift":
            predicted = int(rng.random() >= 0.95)
        else:
            predicted = int(rng.integers(0, 2))
        correct = bool(rng.random() < base_confidence)
        truth = predicted if correct else 1 - predicted
        reported = (
            min(0.995, base_confidence + 0.18)
            if scenario == "overconfident"
            else base_confidence
        )
        posterior = (
            (reported, 1.0 - reported)
            if predicted == 0
            else (1.0 - reported, reported)
        )
        observation = MemoryObservation(
            observation_id=f"{scenario}-{index}",
            namespace=f"audit-{scenario}",
            relation_type="binary",
            source=f"s-{index}",
            target=f"t-{index}",
            posterior=posterior,
            prior=(0.5, 0.5),
            evidence_family=f"event-{index}",
            provenance={"monitor_family": "locked-generator"},
            observed_at=float(index),
        )
        monitor.track(observation)
        monitor.label(
            observation.observation_id,
            str(truth),
            ("0", "1"),
            labeled_at=float(index) + 0.5,
        )


def run(seed: int, scenario: str) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    monitor = CalibrationDriftMonitor(CONFIG)
    add_window(monitor, rng, "stable", 0, REFERENCE_SIZE)
    add_window(monitor, rng, scenario, REFERENCE_SIZE, CURRENT_SIZE)
    report = monitor.report(relation_type="binary")
    codes = sorted(alert["code"] for alert in report["alerts"])
    return {
        "seed": seed,
        "scenario": scenario,
        "status": report["status"],
        "alert_codes": "|".join(codes),
        "nll_delta": report["delta"]["nll"],
        "brier_delta": report["delta"]["brier"],
        "current_ece": report["current"]["ece"],
        "prediction_marginal_js": report["drift"]["prediction_marginal_js"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new output directory; existing paths are never overwritten",
    )
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise SystemExit(
            f"audit output already exists and will not be overwritten: "
            f"{output}"
        )
    rows = [
        run(seed, scenario)
        for scenario in SCENARIOS
        for seed in SEEDS
    ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        summary[scenario] = {
            "seeds": len(selected),
            "alert_rate": float(
                np.mean([row["status"] == "alert" for row in selected])
            ),
            "mean_nll_delta": float(
                np.mean([row["nll_delta"] for row in selected])
            ),
            "mean_brier_delta": float(
                np.mean([row["brier_delta"] for row in selected])
            ),
            "mean_current_ece": float(
                np.mean([row["current_ece"] for row in selected])
            ),
            "mean_prediction_marginal_js": float(
                np.mean(
                    [row["prediction_marginal_js"] for row in selected]
                )
            ),
        }
    payload = {
        "audit": "calibration_drift_monitor_fresh_locked_v2",
        "locked_seeds": list(SEEDS),
        "locked_config": CONFIG.__dict__,
        "success_guards": {
            "stable_alert_rate_max": 0.10,
            "overconfident_alert_rate_min": 0.90,
            "prediction_shift_alert_rate_min": 0.90,
        },
        "summary": summary,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    if summary["stable"]["alert_rate"] > 0.10:
        raise SystemExit("locked guard failed: stable false-alert rate")
    if summary["overconfident"]["alert_rate"] < 0.90:
        raise SystemExit("locked guard failed: overconfidence detection")
    if summary["prediction_shift"]["alert_rate"] < 0.90:
        raise SystemExit("locked guard failed: prediction-shift detection")


if __name__ == "__main__":
    main()
