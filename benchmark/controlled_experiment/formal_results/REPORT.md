# Agent Memory Harness controlled benchmark

## Result

This experiment isolates the memory layer.  A fixed observation generator emits
ambiguous local relation posteriors; no LLM weights or prompts differ between
methods.

- Local relation top-1 accuracy: 64.76%.
- Local true-relation top-2 coverage: 99.98%.
- Cycle-rich hard-label memory node accuracy: 65.59%.
- Cycle-rich posterior memory node accuracy: 94.45%.
- Full posterior minus hard label: +28.86 pp,
  95% CI [+27.89,
  +29.84].
- The same difference without redundant cycles: +0.94 pp,
  95% CI [+0.50,
  +1.35].
- Correct-law minus wrong-order memory: +27.93 pp.
- Correct sample correspondence minus stratified shuffle:
  +77.32 pp.

## Interpretation

The benchmark can establish whether the generic harness improves structured
agent-state memory under controlled ambiguity.  It does not by itself establish
an open-domain LLM gain: the observation encoder is simulated so that memory
inference can be identified separately from language extraction.
