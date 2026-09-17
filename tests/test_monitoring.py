import unittest

from posterior_memory_harness import (
    CalibrationDriftMonitor,
    MemoryObservation,
    MonitorConfig,
    PosteriorMemoryHarness,
    cyclic_law,
)
from posterior_memory_harness.store import InMemoryMemoryStore


def observation(identifier, posterior, *, derived=False, family="encoder-a"):
    return MemoryObservation(
        observation_id=identifier,
        namespace="run",
        relation_type="binary",
        source=f"s-{identifier}",
        target=f"t-{identifier}",
        posterior=tuple(posterior),
        prior=(0.5, 0.5),
        evidence_family=identifier,
        provenance={"monitor_family": family},
        observed_at=float(identifier.split("-")[-1]),
        independent_evidence=not derived,
    )


class CalibrationDriftMonitorTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "reference_size": 4,
            "current_size": 4,
            "min_current_size": 4,
            "ece_bins": 5,
            "max_nll_increase": 0.1,
            "max_brier_increase": 0.05,
            "max_ece": 0.3,
            "max_prediction_js": 0.04,
            "max_tracked": 100,
        }
        values.update(changes)
        return MonitorConfig(**values)

    def populate(self, monitor, current_confidence=0.8, shifted=False):
        predictions = [0, 1, 0, 1]
        truths = [0, 1, 0, 0]
        for offset in range(8):
            predicted = predictions[offset % 4]
            if offset >= 4 and shifted:
                predicted = 0
            confidence = 0.8 if offset < 4 else current_confidence
            posterior = (
                (confidence, 1.0 - confidence)
                if predicted == 0
                else (1.0 - confidence, confidence)
            )
            item = observation(f"obs-{offset}", posterior)
            monitor.track(item)
            truth = truths[offset % 4]
            monitor.label(
                item.observation_id,
                str(truth),
                ("0", "1"),
                labeled_at=item.observed_at + 0.5,
            )

    def test_stable_stream_is_healthy(self):
        monitor = CalibrationDriftMonitor(self.config())
        self.populate(monitor)
        report = monitor.report(relation_type="binary")
        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["alerts"], [])
        self.assertAlmostEqual(report["delta"]["nll"], 0.0)
        self.assertAlmostEqual(report["drift"]["prediction_marginal_js"], 0.0)

    def test_overconfidence_triggers_proper_score_alerts(self):
        monitor = CalibrationDriftMonitor(self.config())
        self.populate(monitor, current_confidence=0.99)
        report = monitor.report(relation_type="binary")
        codes = {alert["code"] for alert in report["alerts"]}
        self.assertEqual(report["status"], "alert")
        self.assertIn("nll_regression", codes)
        self.assertIn("brier_regression", codes)

    def test_prediction_marginal_shift_is_detected(self):
        monitor = CalibrationDriftMonitor(
            self.config(max_prediction_js=0.01)
        )
        self.populate(monitor, shifted=True)
        report = monitor.report(relation_type="binary")
        codes = {alert["code"] for alert in report["alerts"]}
        self.assertIn("prediction_drift", codes)

    def test_delayed_truth_is_immutable_and_derived_records_are_ignored(self):
        monitor = CalibrationDriftMonitor(self.config())
        item = observation("obs-0", (0.7, 0.3))
        self.assertTrue(monitor.track(item))
        self.assertTrue(monitor.label("obs-0", "0", ("0", "1"), labeled_at=1.0))
        self.assertFalse(monitor.label("obs-0", "0", ("0", "1"), labeled_at=2.0))
        with self.assertRaisesRegex(ValueError, "cannot be relabeled"):
            monitor.label("obs-0", "1", ("0", "1"), labeled_at=2.0)
        self.assertFalse(
            monitor.track(
                observation("obs-1", (0.7, 0.3), derived=True)
            )
        )

    def test_harness_tracks_and_reports_by_source_family(self):
        monitor = CalibrationDriftMonitor(self.config())
        harness = PosteriorMemoryHarness(monitor=monitor).register_relation(
            "binary", cyclic_law(2)
        )
        for offset in range(8):
            item = observation(f"obs-{offset}", (0.8, 0.2))
            harness.observe(item)
            harness.record_outcome(
                item.observation_id,
                "0" if offset % 4 != 3 else "1",
                labeled_at=item.observed_at + 0.5,
            )
        report = harness.calibration_report(namespace="run")
        relation = report["relations"]["binary"]
        self.assertEqual(relation["overall"]["counts"]["labeled"], 8)
        self.assertIn("encoder-a", relation["by_source_family"])

    def test_insufficient_data_is_not_reported_as_healthy(self):
        monitor = CalibrationDriftMonitor(self.config())
        item = observation("obs-0", (0.8, 0.2))
        monitor.track(item)
        monitor.label("obs-0", "0", ("0", "1"), labeled_at=1.0)
        report = monitor.report(relation_type="binary")
        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["alerts"], [])

    def test_store_failure_rolls_back_monitor_tracking(self):
        class FailingStore(InMemoryMemoryStore):
            def append(self, item):
                raise RuntimeError("primary store failed")

        monitor = CalibrationDriftMonitor(self.config())
        harness = PosteriorMemoryHarness(
            store=FailingStore(),
            monitor=monitor,
        ).register_relation("binary", cyclic_law(2))
        item = observation("obs-0", (0.8, 0.2))
        with self.assertRaisesRegex(RuntimeError, "primary store failed"):
            harness.observe(item)
        report = monitor.report(relation_type="binary")
        self.assertEqual(report["counts"]["tracked"], 0)

    def test_relation_support_size_cannot_change(self):
        monitor = CalibrationDriftMonitor(self.config())
        monitor.track(observation("obs-0", (0.8, 0.2)))
        incompatible = MemoryObservation(
            observation_id="obs-1",
            namespace="run",
            relation_type="binary",
            source="s",
            target="t",
            posterior=(0.5, 0.3, 0.2),
            prior=(1 / 3, 1 / 3, 1 / 3),
            evidence_family="obs-1",
            observed_at=1.0,
        )
        with self.assertRaisesRegex(ValueError, "support sizes"):
            monitor.track(incompatible)


if __name__ == "__main__":
    unittest.main()
