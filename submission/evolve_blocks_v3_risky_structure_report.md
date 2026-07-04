# Evolve Blocks v3 Risky Structure Report

Evolve Blocks v2 showed very high official validity but no accepted improvement. V3 adds a separate risky structure island that is allowed to create temporarily invalid candidates, then repairs them through fixed-center LP and safety polishing before any official evaluator call.

## What Changed

- `safe_polish_island` keeps the existing LP/refine behavior around strong parents.
- `risky_structure_island` uses destroy-repair operators for small-circle reassignment, gap insertion, boundary refill, and contact-edge breaking.
- `aspect_ratio_island` is Task A specific and sweeps width buckets before repair.
- `gap_graph` ranks candidate empty regions from circle/circle gaps, boundary gaps, and random empty samples.
- Cascade evaluation now records raw validity, pre/post repair violation, repair success, contact graph edit distance, and boundary pattern edit distance.

## Run Summary

- Stop reason: `official_eval_budget`
- Generated programs: `89`
- Novelty rejected: `7`
- Official evaluate calls: `80`
- Accepted improvements: `0`
- Island counts: `{'safe_polish': 34, 'aspect_ratio_island': 7, 'migration': 19, 'risky_structure': 29}`

| Task | Best before | Best after | Improved | Gap to 1.0 | Repair attempted | Repair success | Raw invalid | New contact graphs | New boundary patterns | Best risky delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.999996757107 | 0.999996757107 | False | 3.243e-06 | 44 | 44 | 3 | 36 | 28 | -2.055e-07 |
| B | 0.999997367453 | 0.999997367453 | False | 2.633e-06 | 43 | 43 | 6 | 35 | 32 | -4.187e-08 |

## Risky Operators

| Operator | Attempts | Official evals | Valid | Best delta | Mean delta | Novelty mean | Common failures |
|---|---:|---:|---:|---:|---:|---:|---|
| `aspect_ratio_island` | 4 | 4 | 4 | 0.000e+00 | -7.440e-02 | 0.812 | `{'none': 4}` |
| `boundary_gap_refill` | 5 | 5 | 5 | 0.000e+00 | -4.939e-02 | 0.850 | `{'none': 5}` |
| `contact_edge_break_then_repair` | 7 | 7 | 7 | 0.000e+00 | -4.286e-03 | 0.709 | `{'none': 7}` |
| `destroy_repair_k_small` | 5 | 5 | 5 | 0.000e+00 | -4.912e-02 | 0.850 | `{'none': 5}` |
| `gap_insertion_search` | 4 | 4 | 4 | 0.000e+00 | -5.520e-02 | 0.762 | `{'none': 4}` |
| `small_circle_swap` | 5 | 5 | 5 | 0.000e+00 | -5.239e-02 | 0.810 | `{'none': 5}` |

## Interpretation

- V3 deliberately reduces the bias toward candidates that are valid before repair.
- A candidate still cannot overwrite `solution.py` unless the official evaluator accepts it and its score improves the archive best.
- Official evaluator files and task descriptions remain protected and unchanged.
- The final submitted solution remains static NumPy code with no network, LLM, or external-file dependency.
