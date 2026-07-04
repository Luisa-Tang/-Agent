# Score Breakthrough Search Report

The breakthrough harness searches near public benchmark frontier seeds and current best candidates. It does not modify official evaluators, and final `solution.py` remains static, reproducible, and network-free.

## Public Frontier Seeds

| Source | Availability | Raw objective claim | Local artifacts |
|---|---|---|---|
| ClaudeEvolve circle_packing result | `unavailable_no_local_coordinates` | Not used; no locally available coordinate/code artifact was found. | `[]` |
| DominikKamp/Packing | `coordinates_available_locally` | rectangle/n21 raw sum 2.365832326862653; square/n26 raw sum 2.635983060895661. | `['benchmarks/dominikkamp/rectangle_n21.txt', 'benchmarks/dominikkamp/square_n26.txt']` |
| FICO public Task A solution if coordinates are available | `optional_local_file` | Only used when a local newsolutions.txt Problem 13 coordinate copy is available. | `['benchmarks/fico/SOURCE.md']` |
| OpenEvolve issue #156 code if extractable | `unavailable_no_local_extract` | Not used as a score claim; only local official evaluator results can validate a seed. | `[]` |
| ThetaEvolve circle packing result | `unavailable_no_local_coordinates` | Not used; no locally available coordinate/code artifact was found. | `[]` |

## Task Results

### Task A

- Best candidate: `A_000_benchmark_seed_dominikkamp`
- Best sum_radii: `2.365832327833825`
- Best score: `0.999996757106916`
- Gap to denominator score 1.000000: `3.24289308439862e-06`
- Exceeded denominator: `False`
- Generated candidates: `100`
- Official evaluated candidates: `18`
- Valid official candidates: `18`
- Improvements: `0`

| Verified public seed | Score | Sum radii | Contact graph | Boundary pattern |
|---|---:|---:|---|---|
| `A_90001_public_frontier_dominikkamp` | 0.999996757107 | 2.365832327834 | `736cb82a03549e01` | `left:15,16,19,20|right:0,4,8,11,13|bottom:0,3,9,20|top:5,11,15,17` |

### Task B

- Best candidate: `B_000_benchmark_seed_dominikkamp`
- Best sum_radii: `2.635983060632065`
- Best score: `0.999997367452860`
- Gap to denominator score 1.000000: `2.63254713994687e-06`
- Exceeded denominator: `False`
- Generated candidates: `100`
- Official evaluated candidates: `18`
- Valid official candidates: `18`
- Improvements: `0`

| Verified public seed | Score | Sum radii | Contact graph | Boundary pattern |
|---|---:|---:|---|---|
| `B_90001_public_frontier_dominikkamp` | 0.999997367453 | 2.635983060632 | `8dae3c4e6bea6bd1` | `left:0,11,22,23,25|right:3,9,12,13,16|bottom:3,11,15,20,21|top:2,7,12,19,22` |

## Strategy Contribution

| Strategy family | Official evals | Valid | Best score | Best sum radii | Failure counts |
|---|---:|---:|---:|---:|---|
| `public_frontier_seed` | 2 | 2 | 0.999997367453 | 2.635983060632 | `{'none': 2}` |
| `contact_graph_refinement` | 34 | 34 | 0.999997359453 | 2.635983039546 | `{'none': 34}` |

## Contact Graph Attempts

- Official contact graph evaluations: `34`
- Valid contact graph evaluations: `34`
- Invalid contact graph evaluations: `0`

| Candidate | Task | Score | Sum radii | Contact graph | Boundary pattern |
|---|---|---:|---:|---|---|
| `B_91002_contact_graph_feasibility_refine` | B | 0.999997359453 | 2.635983039546 | `8dae3c4e6bea6bd1` | `left:0,11,22,23,25|right:3,9,12,13,16|bottom:3,11,15,20,21|top:2,7,12,19,22` |
| `B_91003_contact_graph_feasibility_refine` | B | 0.999997359201 | 2.635983038880 | `8dae3c4e6bea6bd1` | `left:0,11,22,23,25|right:3,9,12,13,16|bottom:3,11,15,20,21|top:2,7,12,19,22` |
| `B_91005_contact_graph_feasibility_refine` | B | 0.999997344516 | 2.635983000169 | `8dae3c4e6bea6bd1` | `left:0,11,22,23,25|right:3,9,12,13,16|bottom:3,11,15,20,21|top:2,7,12,19,22` |
| `B_91004_contact_graph_feasibility_refine` | B | 0.999997341469 | 2.635982992140 | `8dae3c4e6bea6bd1` | `left:0,11,22,23,25|right:3,9,12,13,16|bottom:3,11,15,20,21|top:2,7,12,19,22` |
| `B_91007_contact_graph_feasibility_refine` | B | 0.999997339966 | 2.635982988178 | `8dae3c4e6bea6bd1` | `left:0,11,22,23,25|right:3,9,12,13,16|bottom:3,11,15,20,21|top:2,7,12,19,22` |
| `A_91014_contact_graph_feasibility_refine` | A | 0.999996749621 | 2.365832310123 | `736cb82a03549e01` | `left:15,16,19,20|right:0,4,8,11,13|bottom:0,3,9,20|top:5,11,15,17` |
| `A_91012_contact_graph_feasibility_refine` | A | 0.999996748629 | 2.365832307777 | `736cb82a03549e01` | `left:15,16,19,20|right:0,4,8,11,13|bottom:0,3,9,20|top:5,11,15,17` |
| `A_91016_contact_graph_feasibility_refine` | A | 0.999996742399 | 2.365832293038 | `736cb82a03549e01` | `left:15,16,19,20|right:0,4,8,11,13|bottom:0,3,9,20|top:5,11,15,17` |

## Contact Graph Refinement Evidence

- Refinement strategy: `contact_graph_feasibility_refine`.
- Deltas attempted: `[1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6]` across batches.
- Each official-evaluated candidate has a code snapshot, raw evaluator output, contact graph hash, active boundary pattern, and failure type in `agent/archive/metrics/breakthrough_log.jsonl`.
- MAP-Elites-lite buckets are stored in `agent/archive/metrics/novelty_archive.json`.

## Why Evaluators and Final Solutions Stay Safe

- Official `evaluate.py`, `evaluate_all.py`, and task descriptions are not modified.
- Webpage claimed scores are metadata only; only local official evaluator output validates a candidate.
- Current best is not overwritten unless the official evaluator returns a higher score.
- Final `solution.py` files contain static NumPy arrays and no network calls.
- API keys are not written to logs or archive artifacts.
