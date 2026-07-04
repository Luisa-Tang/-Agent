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
- Generated programs: `67`
- Novelty rejected: `5`
- Official evaluate calls: `60`
- Accepted improvements: `0`

| Task | Best before | Best after | Improved | Exceeded 1.0 | Gap to 1.0 | Official evals | Valid official |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.999996757107 | 0.999996757107 | False | False | 3.243e-06 | 33 | 33 |
| B | 0.999997367453 | 0.999997367453 | False | False | 2.633e-06 | 27 | 27 |

## Operator Statistics

| Operator | Attempts | Accepted | Official evals | Valid | Best delta | Mean delta | Avg runtime | Novelty mean | Common failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `aspect_ratio_sweep_local` | 7 | 0 | 7 | 7 | 0.000e+00 | -9.457e-06 | 0.481 | 0.647 | `{'none': 7}` |
| `block_crossover` | 6 | 0 | 5 | 5 | 0.000e+00 | -3.414e-06 | 0.408 | 0.492 | `{'none': 5, 'rejected_novelty': 1}` |
| `boundary_pattern_swap` | 5 | 0 | 5 | 5 | 0.000e+00 | -7.050e-02 | 0.480 | 0.808 | `{'none': 5}` |
| `boundary_slide_mutation` | 8 | 0 | 8 | 8 | 0.000e+00 | -1.375e-05 | 0.482 | 0.677 | `{'none': 8}` |
| `contact_graph_breaking_refine` | 6 | 0 | 3 | 3 | 0.000e+00 | -1.555e-06 | 0.301 | 0.380 | `{'rejected_novelty': 3, 'none': 3}` |
| `contact_graph_preserving_refine` | 7 | 0 | 7 | 7 | 0.000e+00 | -1.911e-05 | 0.480 | 0.670 | `{'none': 7}` |
| `contact_pair_relaxation` | 6 | 0 | 5 | 5 | 0.000e+00 | -1.007e-06 | 0.415 | 0.546 | `{'none': 5, 'rejected_novelty': 1}` |
| `program_patch` | 2 | 0 | 2 | 2 | 0.000e+00 | -1.692e-01 | 0.467 | 0.875 | `{'none': 2}` |
| `radius_group_redistribution` | 8 | 0 | 8 | 8 | 0.000e+00 | -5.506e-05 | 0.484 | 0.759 | `{'none': 8}` |
| `small_circle_reposition` | 3 | 0 | 3 | 3 | 0.000e+00 | -1.392e-01 | 0.480 | 0.833 | `{'none': 3}` |
| `solver_switch` | 7 | 0 | 7 | 7 | 0.000e+00 | -1.537e-05 | 0.498 | 0.634 | `{'none': 7}` |

## Why This Still Matters If No Breakthrough Occurs

A no-improvement run is still useful evidence: it records which program-level operators were tried, which candidates were rejected before expensive official evaluation, how many official evaluations were spent, and why the current static best was preserved. This is stronger evidence than repeatedly jittering coordinates around one contact graph.

## Safety

- Official evaluators and task descriptions are not modified.
- Final `solution.py` files are overwritten only from the best official-valid archive candidate.
- The submitted solutions remain static NumPy code with no network, LLM, or external-file dependency.
- API keys are neither read from prompts nor written to archive, report, or solution artifacts.
