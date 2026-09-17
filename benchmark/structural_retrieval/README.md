# Matched-budget structural retrieval audit

This audit holds the local posterior tensor, observation count, relation law,
and exact inference backend fixed. It changes only which observations survive
the retrieval budget:

- `recent`: newest independent lineages;
- `coverage`: an information-ranked rooted spanning forest, then recent edges;
- `cycle_aware`: the same forest, then information-ranked cycle-closing edges.

Each episode contains an older informative \(S_3\) graph with two independent
cycles and newer prior-like clutter concentrated on one pair of nodes. Every
condition receives exactly five factors. Exact cross-family echoes sharing a
lineage root are then added to verify that repeated representations of one
source event do not move the inferred posterior.

Run a disposable audit from the project root:

```powershell
$env:PYTHONPATH = "src"
python benchmark/structural_retrieval/run_benchmark.py `
  --output tmp/structural-retrieval-audit
```

The output directory is required and must not already exist. The primary
guards require matched selected-factor counts, a positive paired accuracy
interval for cycle-aware versus recent retrieval, a negative paired NLL
interval, and lineage-echo marginal TV no greater than \(10^{-12}\).

## Frozen formal result

The bundled `formal_results/` run uses 20 seeds and 200 episodes per seed:

| Policy | Accuracy | NLL | Brier | Factors |
|---|---:|---:|---:|---:|
| recent | 16.34% | 1.792 | 0.833 | 5 |
| coverage | 81.57% | 0.839 | 0.400 | 5 |
| cycle-aware | 96.27% | 0.277 | 0.103 | 5 |

Cycle-aware minus recent accuracy is +79.93 percentage points with paired
95% seed-bootstrap interval `[78.77, 81.10]`; the NLL interval is
`[-1.519, -1.509]`. Exact lineage echoes have maximum marginal TV `0`.

The stress design intentionally makes recent factors prior-like and
concentrated on one node pair. It is therefore a mechanism test, not an
estimate of expected improvement on natural Agent histories. It also shows
why raw cycle rank is insufficient: recent retrieval retains multigraph
cycles but their cycle-information score is zero.
