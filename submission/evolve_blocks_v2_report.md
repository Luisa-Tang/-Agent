# Evolve Blocks v2 Report

Evolve Blocks v1 produced official-valid candidates but no accepted improvements. The v2 upgrade narrows evolution to named blocks: Task A aspect, geometry, radius, refine, and safety; Task B geometry, radius, refine, and safety.

## What Changed

- Candidate programs now expose named EVOLVE-BLOCK sections.
- Mutation operators target geometry or refine blocks instead of rewriting the whole function.
- Geometry novelty combines code-block novelty, contact graph change, boundary pattern change, center RMSD, radius-distribution change, and strategy-family novelty.
- Parent sampling supports balanced exploit/diverse/risky-novelty modes.
- `block_crossover` combines program blocks rather than splicing final coordinates.

## Run Summary

- Generated programs: `89`
- Novelty rejected: `7`
- Official evaluate calls: `80`
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
| `aspect` | 4 | 1.000 | 0.000e+00 | 0 | 1.000 | 1.000 | 1.061e-01 |
| `geometry` | 74 | 0.919 | 0.000e+00 | 0 | 0.919 | 0.838 | 5.053e-02 |
| `radius` | 7 | 1.000 | 0.000e+00 | 0 | 1.000 | 1.000 | 1.521e-02 |
| `refine` | 6 | 0.833 | 0.000e+00 | 0 | 0.833 | 0.833 | 1.745e-02 |
| `seed` | 2 | 1.000 | 0.000e+00 | 0 | 0.000 | 0.000 | 0.000e+00 |

## Operator Ablation

| Operator | Attempts | Official evals | Valid | Best delta | Mean delta | Novelty mean | Common failures |
|---|---:|---:|---:|---:|---:|---:|---|
| `aspect_ratio_island` | 4 | 4 | 4 | 0.000e+00 | -7.440e-02 | 0.812 | `{'none': 4}` |
| `aspect_ratio_sweep_local` | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 | 0.000 | `{}` |
| `boundary_gap_refill` | 5 | 5 | 5 | 0.000e+00 | -4.939e-02 | 0.850 | `{'none': 5}` |
| `boundary_pattern_swap` | 19 | 17 | 17 | 0.000e+00 | -6.827e-02 | 0.647 | `{'none': 17, 'rejected_novelty': 2}` |
| `boundary_slide_mutation` | 6 | 6 | 6 | 0.000e+00 | -1.107e-02 | 0.667 | `{'none': 6}` |
| `contact_edge_break_then_repair` | 7 | 7 | 7 | 0.000e+00 | -4.286e-03 | 0.709 | `{'none': 7}` |
| `contact_graph_breaking_refine` | 6 | 4 | 4 | 0.000e+00 | -1.505e-03 | 0.522 | `{'none': 4, 'rejected_novelty': 2}` |
| `contact_graph_preserving_refine` | 6 | 5 | 5 | 0.000e+00 | -1.259e-02 | 0.613 | `{'none': 5, 'rejected_novelty': 1}` |
| `contact_pair_relaxation` | 5 | 4 | 4 | 0.000e+00 | -2.512e-02 | 0.578 | `{'none': 4, 'rejected_novelty': 1}` |
| `destroy_repair_k_small` | 5 | 5 | 5 | 0.000e+00 | -4.912e-02 | 0.850 | `{'none': 5}` |
| `gap_insertion_search` | 4 | 4 | 4 | 0.000e+00 | -5.520e-02 | 0.762 | `{'none': 4}` |
| `program_patch` | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 | 0.000 | `{}` |
| `radius_group_redistribution` | 6 | 5 | 5 | 0.000e+00 | -1.500e-03 | 0.649 | `{'none': 5, 'rejected_novelty': 1}` |
| `small_circle_reposition` | 2 | 2 | 2 | 0.000e+00 | -1.895e-01 | 0.875 | `{'none': 2}` |
| `small_circle_swap` | 5 | 5 | 5 | 0.000e+00 | -5.239e-02 | 0.810 | `{'none': 5}` |
| `solver_switch` | 7 | 7 | 7 | 0.000e+00 | -1.570e-03 | 0.722 | `{'none': 7}` |

## Interpretation

- No operator produced a positive delta; closest average result was `aspect_ratio_sweep_local` with mean delta 0.000e+00.
- No block produced a positive delta; strongest geometry novelty came from `aspect` with mean contact-change rate 1.000.
- Final replacement policy did not change: no candidate can overwrite `solution.py` unless official evaluator score improves.
- Official evaluator files and task descriptions are not modified.
