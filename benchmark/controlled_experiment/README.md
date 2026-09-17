# PosteriorGlue Agent Memory Harness

This is a model-agnostic memory middleware prototype.  Observation encoders are
pluggable: an LLM, classifier, NLI model, or verified tool can emit a
`MemoryAtom`.  The core preserves candidate probabilities, applies local-prior
correction, deduplicates evidence families, and performs query-time inference
under a typed relation law.

The controlled benchmark uses noncommutative `S3` state transitions to model
ordered agent checkpoints.  It compares hard-label memory, full posterior
memory, wrong-order inference, stratified shuffled observations, and an oracle
under cycle-free and cycle-rich memory graphs.

```powershell
python test_memory_harness.py
python run_controlled_benchmark.py --outdir results --seeds 20 --episodes-per-seed 200
```

This first benchmark isolates the memory module.  It is not yet an open-domain
LLM benchmark: the local relation posterior is generated in a controlled way so
that improvements can be attributed to memory inference rather than prompting.

## Current controlled result

The fixed 20-seed run is in `formal_results/`.  On cycle-rich memories, full
posterior inference improves node accuracy from 65.59% to 94.45% over hard
top-1 memory: +28.86 percentage points with a paired 95% seed-bootstrap
interval of [27.89, 29.84].  Without redundant cycles the accuracy gain is only
+0.94 points, while NLL and Brier still improve.  Wrong-order and stratified
shuffle controls fail, so the gain requires both the correct relation law and
sample-specific observations.

See `formal_results/REPORT_ZH.md` for the complete interpretation and
`formal_results/PROVENANCE.json` for source/result hashes.
