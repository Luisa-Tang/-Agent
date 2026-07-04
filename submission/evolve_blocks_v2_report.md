# Evolve Blocks v2 Report

Evolve Blocks v1 produced official-valid candidates but no accepted improvements. The v2 upgrade narrows evolution to named blocks: Task A aspect, geometry, radius, refine, and safety; Task B geometry, radius, refine, and safety.

## What Changed

- Candidate programs now expose named EVOLVE-BLOCK sections.
- Mutation operators target geometry or refine blocks instead of rewriting the whole function.
- Geometry novelty combines code-block novelty, contact graph change, boundary pattern change, center RMSD, radius-distribution change, and strategy-family novelty.
- Parent sampling supports balanced exploit/diverse/risky-novelty modes.
- `block_crossover` combines program blocks rather than splicing final coordinates.

## Run Summary

- Generated programs: `67`
- Novelty rejected: `5`
- Official evaluate calls: `60`
- Accepted improvements: `0`
- Block metrics JSON: `agent/archive/evolve/block_metrics.json`
- Block metrics JSONL: `agent/archive/evolve/block_metrics.jsonl`

| Task | Best before | Best after | Improved | Exceeded denominator | Gap to 1.0 |
|---|---:|---:|---:|---:|---:|
| A | 0.999996757107 | 0.999996757107 | False | False | 3.243e-06 |
| B | 0.999997367453 | 0.999997367453 | False | False | 2.633e-06 |

## Block-Level Metrics

| Block changed | Attempts | Valid rate | Best delta | Accepted improvements | Mean contact changed | Mean boundary changed | Mean center RMSD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `aspect` | 7 | 1.000 | 0.000e+00 | 0 | 1.000 | 0.857 | 1.183e-06 |
| `geometry` | 44 | 0.886 | 0.000e+00 | 0 | 0.909 | 0.818 | 2.284e-02 |
| `radius` | 13 | 0.923 | 0.000e+00 | 0 | 1.000 | 0.846 | 9.649e-07 |
| `refine` | 13 | 0.923 | 0.000e+00 | 0 | 1.000 | 0.769 | 5.589e-07 |
| `seed` | 2 | 1.000 | 0.000e+00 | 0 | 0.000 | 0.000 | 0.000e+00 |

## Operator Ablation

| Operator | Attempts | Official evals | Valid | Best delta | Mean delta | Novelty mean | Common failures |
|---|---:|---:|---:|---:|---:|---:|---|
| `aspect_ratio_sweep_local` | 7 | 7 | 7 | 0.000e+00 | -9.457e-06 | 0.647 | `{'none': 7}` |
| `block_crossover` | 6 | 5 | 5 | 0.000e+00 | -3.414e-06 | 0.492 | `{'none': 5, 'rejected_novelty': 1}` |
| `boundary_pattern_swap` | 5 | 5 | 5 | 0.000e+00 | -7.050e-02 | 0.808 | `{'none': 5}` |
| `boundary_slide_mutation` | 8 | 8 | 8 | 0.000e+00 | -1.375e-05 | 0.677 | `{'none': 8}` |
| `contact_graph_breaking_refine` | 6 | 3 | 3 | 0.000e+00 | -1.555e-06 | 0.380 | `{'none': 3, 'rejected_novelty': 3}` |
| `contact_graph_preserving_refine` | 7 | 7 | 7 | 0.000e+00 | -1.911e-05 | 0.670 | `{'none': 7}` |
| `contact_pair_relaxation` | 6 | 5 | 5 | 0.000e+00 | -1.007e-06 | 0.546 | `{'none': 5, 'rejected_novelty': 1}` |
| `program_patch` | 2 | 2 | 2 | 0.000e+00 | -1.692e-01 | 0.875 | `{'none': 2}` |
| `radius_group_redistribution` | 8 | 8 | 8 | 0.000e+00 | -5.506e-05 | 0.759 | `{'none': 8}` |
| `small_circle_reposition` | 3 | 3 | 3 | 0.000e+00 | -1.392e-01 | 0.833 | `{'none': 3}` |
| `solver_switch` | 7 | 7 | 7 | 0.000e+00 | -1.537e-05 | 0.634 | `{'none': 7}` |

## Interpretation

- No operator produced a positive delta; closest average result was `contact_pair_relaxation` with mean delta -1.007e-06.
- No block produced a positive delta; strongest geometry novelty came from `aspect` with mean contact-change rate 1.000.
- Final replacement policy did not change: no candidate can overwrite `solution.py` unless official evaluator score improves.
- Official evaluator files and task descriptions are not modified.
