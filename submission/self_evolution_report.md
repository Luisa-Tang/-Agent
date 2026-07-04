# Self-Evolution Harness Report

GeoEvolve-lite is an optional program-evolution harness. It is inspired by OpenEvolve, ShinkaEvolve, and CodeEvolve concepts, but it does not import those frameworks or replace the stable `agent/run.py` pipeline.

## What Evolves

The harness evolves candidate-generating programs and refinement operators. It does not evolve the final submitted coordinate arrays directly. A child program proposes centers, radii, and optional Task A width/height; the harness repairs and validates the geometry, then emits a static `solution.py` candidate only for official evaluation.

## Program Database

- Program DB: `agent/archive/evolve/program_db.jsonl`
- Program tree: `agent/archive/evolve/program_tree.json`
- Evolve log: `agent/archive/evolve/evolve_log.jsonl`
- Operator stats: `agent/archive/evolve/operator_stats.json`
- Block metrics: `agent/archive/evolve/block_metrics.json`

Each program record stores `program_id`, parent, task, operator, code path/hash, score, sum radii, official evaluator path, contact graph hash, boundary pattern, novelty score, strategy family, timestamp, and metadata.

## Novelty Rejection and Cascade Evaluation

- E0 checks syntax/import and the required `propose_candidate` function.
- E1 executes the generator and runs internal geometry checks.
- E2 computes quick score, contact graph, boundary pattern, code novelty, RMSD, and strategy-family novelty.
- E3 runs official `evaluate.py` only for candidates that pass E0-E2 and the official-evaluation budget.

## Strategy Bandit

Operator selection uses a lightweight UCB-style controller over parameter mutation, solver switch, contact-threshold mutation, program patch fallback, crossover, and depth refinement. The score combines historical improvement, novelty, official validity, runtime penalty, repeated-failure penalty, and exploration bonus.

## Results

- Stop reason: `official_eval_budget`
- Generated programs: `89`
- Novelty rejected: `7`
- Official evaluate calls: `80`
- Accepted improvements: `0`

| Task | Best before | Best after | Improved | Exceeded 1.0 | Gap to 1.0 | Official evals | Valid official |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.999996757107 | 0.999996757107 | False | False | 3.243e-06 | 41 | 41 |
| B | 0.999997367453 | 0.999997367453 | False | False | 2.633e-06 | 39 | 39 |

## Operator Statistics

| Operator | Attempts | Accepted | Official evals | Valid | Best delta | Mean delta | Avg runtime | Novelty mean | Common failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `aspect_ratio_island` | 4 | 0 | 4 | 4 | 0.000e+00 | -7.440e-02 | 0.645 | 0.812 | `{'none': 4}` |
| `aspect_ratio_sweep_local` | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 | 0.000 | 0.000 | `{}` |
| `boundary_gap_refill` | 5 | 0 | 5 | 5 | 0.000e+00 | -4.939e-02 | 0.634 | 0.850 | `{'none': 5}` |
| `boundary_pattern_swap` | 19 | 0 | 17 | 17 | 0.000e+00 | -6.827e-02 | 0.521 | 0.647 | `{'none': 17, 'rejected_novelty': 2}` |
| `boundary_slide_mutation` | 6 | 0 | 6 | 6 | 0.000e+00 | -1.107e-02 | 0.540 | 0.667 | `{'none': 6}` |
| `contact_edge_break_then_repair` | 7 | 0 | 7 | 7 | 0.000e+00 | -4.286e-03 | 0.568 | 0.709 | `{'none': 7}` |
| `contact_graph_breaking_refine` | 6 | 0 | 4 | 4 | 0.000e+00 | -1.505e-03 | 0.427 | 0.522 | `{'none': 4, 'rejected_novelty': 2}` |
| `contact_graph_preserving_refine` | 6 | 0 | 5 | 5 | 0.000e+00 | -1.259e-02 | 0.502 | 0.613 | `{'none': 5, 'rejected_novelty': 1}` |
| `contact_pair_relaxation` | 5 | 0 | 4 | 4 | 0.000e+00 | -2.512e-02 | 0.484 | 0.578 | `{'none': 4, 'rejected_novelty': 1}` |
| `destroy_repair_k_small` | 5 | 0 | 5 | 5 | 0.000e+00 | -4.912e-02 | 0.617 | 0.850 | `{'none': 5}` |
| `gap_insertion_search` | 4 | 0 | 4 | 4 | 0.000e+00 | -5.520e-02 | 0.593 | 0.762 | `{'none': 4}` |
| `program_patch` | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 | 0.000 | 0.000 | `{}` |
| `radius_group_redistribution` | 6 | 0 | 5 | 5 | 0.000e+00 | -1.500e-03 | 0.484 | 0.649 | `{'none': 5, 'rejected_novelty': 1}` |
| `small_circle_reposition` | 2 | 0 | 2 | 2 | 0.000e+00 | -1.895e-01 | 0.509 | 0.875 | `{'none': 2}` |
| `small_circle_swap` | 5 | 0 | 5 | 5 | 0.000e+00 | -5.239e-02 | 0.622 | 0.810 | `{'none': 5}` |
| `solver_switch` | 7 | 0 | 7 | 7 | 0.000e+00 | -1.570e-03 | 0.555 | 0.722 | `{'none': 7}` |

## Why This Still Matters If No Breakthrough Occurs

A no-improvement run is still useful evidence: it records which program-level operators were tried, which candidates were rejected before expensive official evaluation, how many official evaluations were spent, and why the current static best was preserved. This is stronger evidence than repeatedly jittering coordinates around one contact graph.

## Safety

- Official evaluators and task descriptions are not modified.
- Final `solution.py` files are overwritten only from the best official-valid archive candidate.
- The submitted solutions remain static NumPy code with no network, LLM, or external-file dependency.
- API keys are neither read from prompts nor written to archive, report, or solution artifacts.
