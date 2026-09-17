# Locked calibration/drift monitor audit

This fixed audit evaluates the monitor rather than memory-retrieval accuracy.
It uses ten immutable seeds, 400 reference predictions and 400 current
predictions per seed.

The three conditions are:

- `stable`: the calibrated binary generator is unchanged;
- `overconfident`: correctness is unchanged but reported confidence is raised;
- `prediction_shift`: confidence remains calibrated while the predicted-class
  marginal shifts strongly.

The script writes per-seed and aggregate results, then exits non-zero unless
the pre-specified false-alert and detection-rate guards pass:

```powershell
$env:PYTHONPATH = "src"
python benchmark/monitoring_audit/run_locked_audit.py `
  --output tmp/monitoring-audit-rerun
```

The output path is required and must not exist. The runner can therefore never
overwrite the bundled frozen audit directories.

This is a controlled monitor mechanism audit, not a real-LLM memory benchmark.

## Audit history

The first threshold audit used seeds 1100--1109 and a prediction-JS threshold
of 0.03. It passed the stable-stream and overconfidence guards but detected
only 5/10 prediction shifts. Those files are retained under
`development_results_v1/`; the failure was not overwritten.

That run was then treated as development evidence. The fresh v2 configuration
locked a JS threshold of 0.01 and new seeds 2100--2109 before execution. Its
outputs are written once to `fresh_locked_results_v2/`.
