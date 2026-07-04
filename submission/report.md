# AlgorithmOptimization Local Agent Report

## 1. Overview

This repository contains a deterministic local Agent system that generates `solution.py` candidates for two circle-packing optimization tasks, runs the official evaluators through subprocesses, parses feedback, archives every attempt, and exports the best valid candidates. The harness follows an observe -> think -> act -> observe loop so every code-evolution step is traceable.

## 2. System Architecture

- **ProblemParser**: `agent/run.py` reads task descriptions and evaluator files before the loop starts.
- **CandidateGenerator**: `agent/candidate_generators.py` creates standalone solution code from safe grids, staggered starts, SLSQP search, multi-start search, perturb-and-repair, and external benchmark warm-start seeds.
- **EvaluatorAdapter**: `agent/evaluator_adapter.py` writes candidates into the official task directories and runs `evaluate.py` as the source of truth.
- **FeedbackReflector**: `agent/run.py` maps failures and plateau behavior to the next strategy; `agent/llm_reflector.py` can optionally ask a compatible Chat Completions endpoint for a strategy suggestion.
- **ArchiveManager**: `agent/archive.py` stores metadata, raw evaluator output, code snapshots, and best valid candidates.
- **GeoEvolve-lite**: optional `agent/evolve/` harness evolves candidate-generating programs, novelty filters them, runs cascade evaluation, and exports only official-valid improvements.
- **Exporter / Reporter**: `agent/run.py` exports final solutions and `agent/report_data.py` creates this report.
- **Specialist tools**: `agent/skills/` contains project-local reusable procedures for SLSQP search, repair, evaluator feedback, static export, and archive observability.

### Workflow vs Agent

The workflow is the fixed reproducible harness: parse context, generate a candidate, run official evaluators, archive, and export. The Agent layer is the decision policy on top of that workflow: it observes evaluator/archives, chooses a specialist strategy, acts by generating code, and observes official feedback before continuing.

### Observe -> Think -> Act -> Observe Loop

- **Observe**: read the previous evaluator result, best archive state, no-improvement count, and strategy statistics.
- **Think**: select a strategy with deterministic policy or optional LLM reflection constrained to the strategy whitelist.
- **Act**: generate standalone `solution.py` code using a specialist optimizer/repair/export operation.
- **Observe**: run the official evaluator, classify failure, compute geometry metrics, and write the trace to JSONL/logs.

### Manager + Specialist Tools Pattern

`agent/run.py` is the manager. It delegates to specialist tools: SLSQP candidate generation, fixed-center LP repair, evaluator feedback parsing, archive/statistics memory, and static solution export. The evaluator governs code evolution: only candidates accepted by official scripts can become final exports.

### Skill-Based Reusable Procedures

The `agent/skills/` layer contains reusable procedures, not personas: `packing-slsqp`, `packing-repair`, `evaluator-feedback`, `static-export`, and `archive-observability`. The manager records which procedures were consulted in each iteration through `skills_used`, so the report can audit what specialist workflow shaped each candidate.

### LangGraph Orchestration Layer

`agent/langgraph_runner.py` is an optional orchestration entrypoint. It wraps the existing deterministic pipeline in a LangGraph `StateGraph` without replacing `agent/run.py`.
State / nodes / conditional edges:
- State: `GeoOptState` records task, iteration, archive summary, strategy stats, selected strategy, evaluator result, best candidate, skills, and artifacts.
- Nodes: `load_task`, `observe_archive`, `select_strategy`, `generate_candidate`, `evaluate_candidate`, `parse_feedback`, `update_archive`, `static_export`, and `safety_check`.
- Conditional edges: after `update_archive`, the graph routes to portfolio selection, direct repair/generation, static export, safety check, or END based on iteration budget, evaluator failure, benchmark availability, plateau, and valid improvement.
Why this is not role-play multi-agent: the graph is one Agent state machine with explicit nodes and deterministic module calls; it does not introduce persona prompts or separate role-playing agents.
Reuse: the LangGraph runner calls the same `EvaluatorAdapter`, `CandidateGenerator`, `ArchiveManager`, `StrategyPortfolioController`, and `SafetyGuard` modules used by the stable pipeline.
Fallback: `agent/run.py` does not import LangGraph and remains runnable when the optional dependency is missing.
- Mermaid graph: `submission/demo/agent_graph.mmd` exists `True`.
- Text graph: `submission/demo/agent_graph.txt` exists `True`.
- LangGraph node log: `agent/archive/metrics/langgraph_run_log.jsonl` records `0` node executions.

### Explicit Agent State Graph

Each iteration is recorded as `observe -> decide -> act -> evaluate -> archive` using `AgentState` snapshots.
| Phase | Recorded snapshots |
|---|---:|
| observe | 10 |
| decide | 10 |
| act | 10 |
| evaluate | 10 |
| archive | 10 |

Example state paths: `A_000_benchmark_seed_dominikkamp`: observe -> decide -> act -> evaluate -> archive; `A_001_hexagonal_or_staggered_initialization`: observe -> decide -> act -> evaluate -> archive; `A_002_scipy_slsqp_joint`: observe -> decide -> act -> evaluate -> archive

### Strategy Portfolio Controller

The controller scores strategies from archive history, evaluator failures, plateau state, score gap, and remaining budget.
| Strategy | Attempts | Validity rate | Best score | Avg score delta | Avg runtime | Common failures | Last used |
|---|---:|---:|---:|---:|---:|---|---:|
| benchmark_seed_dominikkamp | 2 | 1.000 | 0.999997 | 0.999997 | 0.180 | `{'none': 2}` | 0.000000000 |
| hexagonal_or_staggered_initialization | 2 | 1.000 | 0.915143 | -0.129478 | 0.185 | `{'low_score': 1, 'none': 1}` | 1.000000000 |
| perturb_best_and_repair | 4 | 1.000 | 0.999997 | -8.72406e-06 | 0.185 | `{'plateau': 4}` | 4.000000000 |
| scipy_slsqp_joint | 2 | 1.000 | 0.985362 | -0.0153804 | 0.175 | `{'none': 2}` | 2.000000000 |
| self_evolve_contact_threshold_mutation | 2 | 1.000 | 0.999997 | -3.33527e-08 | 0.000 | `{'none': 2}` | 99013.000000000 |
| self_evolve_crossover | 7 | 1.000 | 0.999997 | -3.99546e-06 | 0.000 | `{'none': 7}` | 99023.000000000 |
| self_evolve_depth_refinement | 4 | 1.000 | 0.999997 | -5.11694e-07 | 0.000 | `{'none': 4}` | 99017.000000000 |
| self_evolve_parameter_mutation | 10 | 1.000 | 0.999997 | -1.99343e-05 | 0.000 | `{'none': 10}` | 99022.000000000 |
| self_evolve_program_patch | 9 | 1.000 | 0.999997 | -2.64865e-06 | 0.000 | `{'none': 9}` | 99024.000000000 |
| self_evolve_solver_switch | 8 | 1.000 | 0.999996 | -1.20086e-05 | 0.000 | `{'none': 8}` | 99024.000000000 |

### Execution Lineage and Replay

Best-candidate lineage DAGs are emitted as replayable JSON. Each node includes parent, strategy, input/output artifacts, code hash, data hash, official score, and decision reason.
- Task A: `agent/archive/lineage/task_A_best_lineage.json` best `A_000_benchmark_seed_dominikkamp`, nodes `25`, chain length `1`.
- Task B: `agent/archive/lineage/task_B_best_lineage.json` best `B_000_benchmark_seed_dominikkamp`, nodes `25`, chain length `1`.

### Safety Guard and Protected Files

- Overall safety status: `True`
- Protected files unchanged: `True`
- Protected git diff entries: `[]`
- API key pattern matches: `0`
- `task_A/solution.py` imports `['numpy']`; network matches `[]`; passed `True`.
- `task_B/solution.py` imports `['numpy']`; network matches `[]`; passed `True`.

### Skill Usage Statistics

- Loaded skills: `['archive-observability', 'evaluator-feedback', 'packing-repair', 'packing-slsqp', 'static-export']`
- Iteration records: `10`
| Skill | Uses |
|---|---:|
| archive-observability | 10 |
| evaluator-feedback | 10 |
| packing-repair | 6 |
| packing-slsqp | 2 |
| static-export | 2 |

### Human-Agent Division Audit

Human contributions and Agent actions are audited with evidence artifacts.
- Human-provided items: `3`
- Agent-completed items: `4`
- Audit files: `submission/human_agent_division.md`, `submission/human_agent_division.json`

## 3. Code Generation Strategy

The Agent uses template-based generation. Each candidate is a complete Python module with literal NumPy arrays for centers and radii, plus a small safety repair routine. Candidate search combines conservative grid layouts, hexagonal/staggered initializations, SLSQP joint optimization over centers and radii, multi-start SLSQP, and local perturbation around the best valid candidate. It can also convert tracked public benchmark geometry into static candidates and submit them to the same official evaluator path. For fixed centers, the Agent solves a linear program to maximize radii under boundary and pairwise non-overlap constraints, then applies a tiny final shrink/repair.

LLM use is optional. When `--use-llm` is passed and `DEEPSEEK_API_KEY` or a compatible fallback key is set, the Agent asks the configured Chat Completions-compatible endpoint to choose among the existing deterministic strategies. The LLM is not allowed to modify official evaluators, and final exported `solution.py` files remain standalone with no network or LLM dependency.

## 4. External Benchmark Warm-start

The Agent can use public DominikKamp/Packing geometry files as external benchmark warm-start candidates. This is a seed source inside the Agent search space, not hidden data and not manually hand-written coordinates. Each converted seed is emitted as a standalone `solution.py` candidate, then accepted or rejected only by the official `evaluate.py` scripts.

- Source: https://github.com/DominikKamp/Packing
- Task B seed: `benchmarks/dominikkamp/square_n26.txt` from `square/n26/circlepacking_n26.txt`
- Task A seed: `benchmarks/dominikkamp/rectangle_n21.txt` from `rectangle/n21/rectangle_n21.txt`

| Task | Candidate | Source file | Raw sum radii | Official valid | Official score | Official sum radii | Decision |
|---|---|---|---:|---:|---:|---:|---|
| A | `A_000_benchmark_seed_dominikkamp` | `rectangle/n21/rectangle_n21.txt` | 2.365832326863 | True | 0.999997 | 2.365832 | Archive improved; next step can exploit this candidate with perturbation or diversify with multi-start. |
| B | `B_000_benchmark_seed_dominikkamp` | `square/n26/circlepacking_n26.txt` | 2.635983060896 | True | 0.999997 | 2.635983 | Archive improved; next step can exploit this candidate with perturbation or diversify with multi-start. |

### Benchmark-Neighborhood Refinement

After a public benchmark seed is available, the Agent can run three small neighborhood refinements: fixed-center radius LP, micro center/width perturbation followed by radius LP, and an optional FICO Problem 13 Task A seed if a local public copy is available. These are lightweight local candidate generators, not a new framework. Every candidate still goes through the official evaluator before it can replace the best valid archive entry.

No benchmark-neighborhood refinement candidate was evaluated in this run.

### Score Breakthrough Harness

The optional breakthrough harness searches near public frontier seeds and contact graph neighborhoods without modifying official evaluators.
- Detailed report: `submission/breakthrough_report.md`
- Candidate log: `agent/archive/metrics/breakthrough_log.jsonl`
- Novelty archive: `agent/archive/metrics/novelty_archive.json`
| Task | Best score | Best sum radii | Gap to 1.0 | Exceeded 1.0 | Generated | Official evals | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.999996757107 | 2.365832327834 | 3.243e-06 | False | 100 | 18 | 18 |
| B | 0.999997367453 | 2.635983060632 | 2.633e-06 | False | 100 | 18 | 18 |

### Self-Evolution Harness

The optional GeoEvolve-lite harness follows OpenEvolve/ShinkaEvolve/CodeEvolve-inspired ideas without importing those frameworks: a program database, EVOLVE-BLOCK mutations, novelty rejection, cascade evaluation, and an operator bandit.
- Detailed report: `submission/self_evolution_report.md`
- Program DB: `agent/archive/evolve/program_db.jsonl`
- Program tree: `agent/archive/evolve/program_tree.json`
- Evolve log: `agent/archive/evolve/evolve_log.jsonl`
- Operator stats: `agent/archive/evolve/operator_stats.json`

Why programs rather than coordinates: the evolved artifact is a small `propose_candidate(parent, rng, context)` generator/refinement operator. Its output is converted to a static candidate and must pass official `evaluate.py` before it can affect final export.

| Task | Best before | Best after | Improved | Exceeded 1.0 | Gap to 1.0 | Official evals | Valid official |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.999996757107 | 0.999996757107 | False | False | 3.243e-06 | 20 | 20 |
| B | 0.999997367453 | 0.999997367453 | False | False | 2.633e-06 | 20 | 20 |

- Generated programs: `50`
- Novelty rejected: `8`
- Official evaluate calls: `40`
- Accepted improvements: `0`

| Operator | Attempts | Valid | Best delta | Novelty mean | Common failures |
|---|---:|---:|---:|---:|---|
| `contact_threshold_mutation` | 5 | 2 | 0.000e+00 | 0.200 | `{'none': 2, 'rejected_novelty': 3}` |
| `crossover` | 8 | 7 | 0.000e+00 | 0.456 | `{'none': 7, 'rejected_novelty': 1}` |
| `depth_refinement` | 7 | 4 | 0.000e+00 | 0.322 | `{'none': 4, 'rejected_novelty': 3}` |
| `parameter_mutation` | 10 | 10 | 0.000e+00 | 0.635 | `{'none': 10}` |
| `program_patch` | 9 | 9 | 0.000e+00 | 0.552 | `{'none': 9}` |
| `solver_switch` | 9 | 8 | 0.000e+00 | 0.587 | `{'none': 8, 'rejected_novelty': 1}` |

## 5. Feedback Utilization

Evaluator output is parsed for score, `sum_radii`, validity, and failure type. Overlap failures trigger more conservative repair. Outside-boundary failures trigger boundary-tight generation. Low but valid scores move the Agent toward multi-start and structured initializations. Plateaued valid runs trigger perturb-and-repair around the current best candidate.

Failure classification uses the explicit taxonomy: `shape_error`, `nonfinite`, `negative_radius`, `perimeter_error`, `boundary_violation`, `overlap`, `timeout`, `low_score`, `plateau`, and `unknown`.

## 6. Termination and Decision Mechanism

The loop terminates after the configured iteration budget or time budget. The archive keeps every candidate, but only official-evaluator-valid candidates can become final exports. If no valid optimized candidate is available, the deterministic safe grid fallback remains valid.

## 7. Results

- Task A best candidate: `A_000_benchmark_seed_dominikkamp`
- Task A sum_radii: `2.365832`
- Task A score: `0.999997` using denominator `2.365840`
- Task A width/height: `0.976731108` / `1.023268892`
- Task B best candidate: `B_000_benchmark_seed_dominikkamp`
- Task B sum_radii: `2.635983`
- Task B score: `0.999997` using denominator `2.635990`
- Combined best-valid score: `0.999997`

### Iteration Trajectory

| Task | Iteration | Candidate | Strategy | Valid | Score | Sum radii | Failure |
|---|---:|---|---|---:|---:|---:|---|
| A | 0 | `A_000_benchmark_seed_dominikkamp` | benchmark_seed_dominikkamp | True | 0.999997 | 2.365832 | none |
| A | 1 | `A_001_hexagonal_or_staggered_initialization` | hexagonal_or_staggered_initialization | True | 0.825895 | 1.953936 | low_score |
| A | 2 | `A_002_scipy_slsqp_joint` | scipy_slsqp_joint | True | 0.985362 | 2.331209 | none |
| A | 3 | `A_003_perturb_best_and_repair` | perturb_best_and_repair | True | 0.999996 | 2.365831 | plateau |
| A | 4 | `A_004_perturb_best_and_repair` | perturb_best_and_repair | True | 0.999996 | 2.365831 | plateau |
| B | 0 | `B_000_benchmark_seed_dominikkamp` | benchmark_seed_dominikkamp | True | 0.999997 | 2.635983 | none |
| B | 1 | `B_001_hexagonal_or_staggered_initialization` | hexagonal_or_staggered_initialization | True | 0.915143 | 2.412309 | none |
| B | 2 | `B_002_scipy_slsqp_joint` | scipy_slsqp_joint | True | 0.983871 | 2.593475 | none |
| B | 3 | `B_003_perturb_best_and_repair` | perturb_best_and_repair | True | 0.999964 | 2.635894 | plateau |
| B | 4 | `B_004_perturb_best_and_repair` | perturb_best_and_repair | True | 0.999997 | 2.635982 | plateau |
| A | 99005 | `A_evolve_A_P_00003` | self_evolve_parameter_mutation | True | 0.999924 | 2.365660 | none |
| A | 99006 | `A_evolve_A_P_00004` | self_evolve_solver_switch | True | 0.999988 | 2.365812 | none |
| A | 99007 | `A_evolve_A_P_00005` | self_evolve_contact_threshold_mutation | True | 0.999997 | 2.365832 | none |
| A | 99008 | `A_evolve_A_P_00006` | self_evolve_program_patch | True | 0.999996 | 2.365831 | none |
| B | 99005 | `B_evolve_B_P_00007` | self_evolve_crossover | True | 0.999997 | 2.635983 | none |
| B | 99006 | `B_evolve_B_P_00008` | self_evolve_depth_refinement | True | 0.999997 | 2.635983 | none |
| B | 99007 | `B_evolve_B_P_00009` | self_evolve_parameter_mutation | True | 0.999989 | 2.635961 | none |
| B | 99008 | `B_evolve_B_P_00010` | self_evolve_solver_switch | True | 0.999996 | 2.635981 | none |
| A | 99009 | `A_evolve_A_P_00011` | self_evolve_program_patch | True | 0.999989 | 2.365814 | none |
| A | 99010 | `A_evolve_A_P_00012` | self_evolve_depth_refinement | True | 0.999996 | 2.365830 | none |
| A | 99011 | `A_evolve_A_P_00014` | self_evolve_crossover | True | 0.999990 | 2.365816 | none |
| B | 99009 | `B_evolve_B_P_00015` | self_evolve_parameter_mutation | True | 0.999976 | 2.635926 | none |
| B | 99010 | `B_evolve_B_P_00016` | self_evolve_solver_switch | True | 0.999976 | 2.635926 | none |
| B | 99011 | `B_evolve_B_P_00017` | self_evolve_program_patch | True | 0.999996 | 2.635980 | none |
| B | 99012 | `B_evolve_B_P_00018` | self_evolve_crossover | True | 0.999995 | 2.635978 | none |
| A | 99012 | `A_evolve_A_P_00019` | self_evolve_depth_refinement | True | 0.999997 | 2.365832 | none |
| A | 99013 | `A_evolve_A_P_00020` | self_evolve_parameter_mutation | True | 0.999994 | 2.365826 | none |
| A | 99014 | `A_evolve_A_P_00021` | self_evolve_solver_switch | True | 0.999996 | 2.365831 | none |
| A | 99015 | `A_evolve_A_P_00022` | self_evolve_program_patch | True | 0.999997 | 2.365832 | none |
| B | 99013 | `B_evolve_B_P_00023` | self_evolve_contact_threshold_mutation | True | 0.999997 | 2.635983 | none |
| B | 99014 | `B_evolve_B_P_00024` | self_evolve_crossover | True | 0.999997 | 2.635982 | none |
| B | 99015 | `B_evolve_B_P_00025` | self_evolve_parameter_mutation | True | 0.999996 | 2.635981 | none |
| A | 99016 | `A_evolve_A_P_00028` | self_evolve_program_patch | True | 0.999994 | 2.365826 | none |
| A | 99017 | `A_evolve_A_P_00029` | self_evolve_parameter_mutation | True | 0.999996 | 2.365832 | none |
| B | 99016 | `B_evolve_B_P_00031` | self_evolve_crossover | True | 0.999979 | 2.635935 | none |
| B | 99017 | `B_evolve_B_P_00032` | self_evolve_depth_refinement | True | 0.999996 | 2.635981 | none |
| B | 99018 | `B_evolve_B_P_00033` | self_evolve_program_patch | True | 0.999997 | 2.635983 | none |
| B | 99019 | `B_evolve_B_P_00034` | self_evolve_solver_switch | True | 0.999961 | 2.635888 | none |
| A | 99018 | `A_evolve_A_P_00035` | self_evolve_parameter_mutation | True | 0.999996 | 2.365830 | none |
| A | 99019 | `A_evolve_A_P_00036` | self_evolve_crossover | True | 0.999997 | 2.365832 | none |
| A | 99020 | `A_evolve_A_P_00037` | self_evolve_solver_switch | True | 0.999971 | 2.365773 | none |
| B | 99020 | `B_evolve_B_P_00039` | self_evolve_program_patch | True | 0.999997 | 2.635982 | none |
| B | 99021 | `B_evolve_B_P_00040` | self_evolve_parameter_mutation | True | 0.999997 | 2.635982 | none |
| A | 99021 | `A_evolve_A_P_00043` | self_evolve_solver_switch | True | 0.999996 | 2.365831 | none |
| A | 99022 | `A_evolve_A_P_00044` | self_evolve_parameter_mutation | True | 0.999906 | 2.365617 | none |
| A | 99023 | `A_evolve_A_P_00045` | self_evolve_program_patch | True | 0.999989 | 2.365813 | none |
| A | 99024 | `A_evolve_A_P_00046` | self_evolve_solver_switch | True | 0.999994 | 2.365826 | none |
| B | 99022 | `B_evolve_B_P_00047` | self_evolve_parameter_mutation | True | 0.999997 | 2.635982 | none |
| B | 99023 | `B_evolve_B_P_00049` | self_evolve_crossover | True | 0.999997 | 2.635982 | none |
| B | 99024 | `B_evolve_B_P_00050` | self_evolve_program_patch | True | 0.999995 | 2.635976 | none |

### Skill Usage Summary

| Skill | Iteration uses |
|---|---:|
| archive-observability | 50 |
| evaluator-feedback | 50 |
| packing-repair | 46 |
| packing-slsqp | 2 |
| static-export | 42 |

### Strategy Archive Statistics

| Strategy | Attempts | Validity rate | Best score | Avg score improvement |
|---|---:|---:|---:|---:|
| benchmark_seed_dominikkamp | 2 | 1.000 | 0.999997 | 0.999997 |
| hexagonal_or_staggered_initialization | 2 | 1.000 | 0.915143 | -0.129478 |
| perturb_best_and_repair | 4 | 1.000 | 0.999997 | -0.000009 |
| scipy_slsqp_joint | 2 | 1.000 | 0.985362 | -0.015380 |
| self_evolve_contact_threshold_mutation | 2 | 1.000 | 0.999997 | -0.000000 |
| self_evolve_crossover | 7 | 1.000 | 0.999997 | -0.000004 |
| self_evolve_depth_refinement | 4 | 1.000 | 0.999997 | -0.000001 |
| self_evolve_parameter_mutation | 10 | 1.000 | 0.999997 | -0.000020 |
| self_evolve_program_patch | 9 | 1.000 | 0.999997 | -0.000003 |
| self_evolve_solver_switch | 8 | 1.000 | 0.999996 | -0.000012 |

### Best Geometry Safety Metrics

| Task | Min pairwise margin | Min boundary margin | Sum radii | Score | Width | Height |
|---|---:|---:|---:|---:|---:|---:|
| A | 7.358e-10 | 1.240e-11 | 2.365832 | 0.999997 | 0.976731108 | 1.023268892 |
| B | 3.766e-10 | 2.328e-10 | 2.635983 | 0.999997 |  |  |

### Final Evaluator Output

```text
============================================================
  Circle Packing in Rectangle  (n=21)
  File : /home/wuyou/projects/AlgorithmOptimization/task_A/solution.py
============================================================
  Elapsed : 0.18s
  sum_radii : 2.365832
  Target    : 2.365840
  Score     : 0.999997
============================================================


============================================================
  Circle Packing in Unit Square  (n=26)
  File : /home/wuyou/projects/AlgorithmOptimization/task_B/solution.py
============================================================
  Elapsed : 0.19s
  sum_radii : 2.635983
  Target    : 2.635990
  Score     : 0.999997
============================================================

============================================================
  Final Score
============================================================
  task_A (Circle Packing in Rectangle)   :  0.999997
  task_B (Circle Packing in Unit Square) :  0.999997
  Combined                               :  0.999997
============================================================
```

## 8. Human-Agent Division

The human provided the high-level system design, constraints, required interfaces, and quality bar. The Agent implemented the framework, ran the local evaluator loop, generated candidate solution files, selected the best valid candidates, exported artifacts, and produced logs. Optional LLM connectivity is supported but the recorded run may use deterministic fallback unless `--use-llm` and credentials are provided. The main manual assumption was using the local conda Python environment when the system `python` command was unavailable.

## 9. Limitations and Future Work

- Add broader global search and population-based evolution.
- Add more diverse initialization families and symmetry-breaking operators.
- Use LLM-guided operator generation when API access is configured, while keeping deterministic fallback.
- Add visualization-based diagnosis for overlap and wasted-space patterns.
- Run longer non-fast searches for higher scores.

## PDF Conversion

If a PDF is required, convert this Markdown file with a local tool such as:

```bash
pandoc submission/report.md -o submission/report.pdf
```
