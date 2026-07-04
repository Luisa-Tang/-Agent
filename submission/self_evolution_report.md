# Self-Evolution Harness Report

GeoEvolve-lite is an optional program-evolution harness. It is inspired by OpenEvolve, ShinkaEvolve, and CodeEvolve concepts, but it does not import those frameworks or replace the stable `agent/run.py` pipeline.

## What Evolves

The harness evolves candidate-generating programs and refinement operators. It does not evolve the final submitted coordinate arrays directly. A child program proposes centers, radii, and optional Task A width/height; the harness repairs and validates the geometry, then emits a static `solution.py` candidate only for official evaluation.

## Program Database

- Program DB: `agent/archive/evolve/program_db.jsonl`
- Program tree: `agent/archive/evolve/program_tree.json`
- Evolve log: `agent/archive/evolve/evolve_log.jsonl`
- Operator stats: `agent/archive/evolve/operator_stats.json`

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
- Generated programs: `50`
- Novelty rejected: `8`
- Official evaluate calls: `40`
- Accepted improvements: `0`

| Task | Best before | Best after | Improved | Exceeded 1.0 | Gap to 1.0 | Official evals | Valid official |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.999996757107 | 0.999996757107 | False | False | 3.243e-06 | 20 | 20 |
| B | 0.999997367453 | 0.999997367453 | False | False | 2.633e-06 | 20 | 20 |

## Operator Statistics

| Operator | Attempts | Accepted | Official evals | Valid | Best delta | Mean delta | Avg runtime | Novelty mean | Common failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `contact_threshold_mutation` | 5 | 0 | 2 | 2 | 0.000e+00 | -1.334e-08 | 0.199 | 0.200 | `{'rejected_novelty': 3, 'none': 2}` |
| `crossover` | 8 | 0 | 7 | 7 | 0.000e+00 | -3.496e-06 | 0.386 | 0.456 | `{'none': 7, 'rejected_novelty': 1}` |
| `depth_refinement` | 7 | 0 | 4 | 4 | 0.000e+00 | -2.924e-07 | 0.279 | 0.322 | `{'none': 4, 'rejected_novelty': 3}` |
| `parameter_mutation` | 10 | 0 | 10 | 10 | 0.000e+00 | -1.993e-05 | 0.447 | 0.635 | `{'none': 10}` |
| `program_patch` | 9 | 0 | 9 | 9 | 0.000e+00 | -2.649e-06 | 0.433 | 0.552 | `{'none': 9}` |
| `solver_switch` | 9 | 0 | 8 | 8 | 0.000e+00 | -1.067e-05 | 0.395 | 0.587 | `{'none': 8, 'rejected_novelty': 1}` |

## Why This Still Matters If No Breakthrough Occurs

A no-improvement run is still useful evidence: it records which program-level operators were tried, which candidates were rejected before expensive official evaluation, how many official evaluations were spent, and why the current static best was preserved. This is stronger evidence than repeatedly jittering coordinates around one contact graph.

## Safety

- Official evaluators and task descriptions are not modified.
- Final `solution.py` files are overwritten only from the best official-valid archive candidate.
- The submitted solutions remain static NumPy code with no network, LLM, or external-file dependency.
- API keys are neither read from prompts nor written to archive, report, or solution artifacts.
