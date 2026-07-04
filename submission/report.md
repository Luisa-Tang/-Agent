# AlgorithmOptimization Local Agent Report

## 1. Overview

This repository contains a deterministic local Agent system that generates `solution.py` candidates for two circle-packing optimization tasks, runs the official evaluators through subprocesses, parses feedback, archives every attempt, and exports the best valid candidates. The harness follows an observe -> think -> act -> observe loop so every code-evolution step is traceable.

## 2. System Architecture

- **ProblemParser**: `agent/run.py` reads task descriptions and evaluator files before the loop starts.
- **CandidateGenerator**: `agent/candidate_generators.py` creates standalone solution code from safe grids, staggered starts, SLSQP search, multi-start search, perturb-and-repair, and external benchmark warm-start seeds.
- **EvaluatorAdapter**: `agent/evaluator_adapter.py` writes candidates into the official task directories and runs `evaluate.py` as the source of truth.
- **FeedbackReflector**: `agent/run.py` maps failures and plateau behavior to the next strategy; `agent/llm_reflector.py` can optionally ask a compatible Chat Completions endpoint for a strategy suggestion.
- **ArchiveManager**: `agent/archive.py` stores metadata, raw evaluator output, code snapshots, and best valid candidates.
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
| benchmark_seed_dominikkamp | 2 | 1.000 | 0.999997 | 0.999997 | 0.185 | `{'none': 2}` | 0.000000000 |
| contact_graph_feasibility_refine | 34 | 1.000 | 0.999997 | -0.0209246 | 0.000 | `{'none': 34}` | 91018.000000000 |
| hexagonal_or_staggered_initialization | 2 | 1.000 | 0.915143 | -0.129478 | 0.190 | `{'low_score': 1, 'none': 1}` | 1.000000000 |
| perturb_best_and_repair | 4 | 1.000 | 0.999997 | -8.72406e-06 | 0.175 | `{'plateau': 4}` | 4.000000000 |
| public_frontier_dominikkamp | 2 | 1.000 | 0.999997 | 0 | 0.000 | `{'none': 2}` | 90001.000000000 |
| scipy_slsqp_joint | 2 | 1.000 | 0.985362 | -0.0153804 | 0.190 | `{'none': 2}` | 2.000000000 |

### Execution Lineage and Replay

Best-candidate lineage DAGs are emitted as replayable JSON. Each node includes parent, strategy, input/output artifacts, code hash, data hash, official score, and decision reason.
- Task A: `agent/archive/lineage/task_A_best_lineage.json` best `A_000_benchmark_seed_dominikkamp`, nodes `23`, chain length `1`.
- Task B: `agent/archive/lineage/task_B_best_lineage.json` best `B_000_benchmark_seed_dominikkamp`, nodes `23`, chain length `1`.

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
| A | 90001 | `A_90001_public_frontier_dominikkamp` | public_frontier_dominikkamp | True | 0.999997 | 2.365832 | none |
| A | 91002 | `A_91002_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91003 | `A_91003_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91004 | `A_91004_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91005 | `A_91005_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91006 | `A_91006_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.985362 | 2.331209 | none |
| A | 91007 | `A_91007_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91008 | `A_91008_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.985362 | 2.331209 | none |
| A | 91009 | `A_91009_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.825896 | 1.953938 | none |
| A | 91010 | `A_91010_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.985362 | 2.331209 | none |
| A | 91011 | `A_91011_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.825896 | 1.953938 | none |
| A | 91012 | `A_91012_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91013 | `A_91013_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.825896 | 1.953938 | none |
| A | 91014 | `A_91014_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91015 | `A_91015_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91016 | `A_91016_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91017 | `A_91017_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| A | 91018 | `A_91018_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.365832 | none |
| B | 90001 | `B_90001_public_frontier_dominikkamp` | public_frontier_dominikkamp | True | 0.999997 | 2.635983 | none |
| B | 91002 | `B_91002_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.635983 | none |
| B | 91003 | `B_91003_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.635983 | none |
| B | 91004 | `B_91004_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.635983 | none |
| B | 91005 | `B_91005_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.635983 | none |
| B | 91006 | `B_91006_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999964 | 2.635895 | none |
| B | 91007 | `B_91007_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999997 | 2.635983 | none |
| B | 91008 | `B_91008_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999964 | 2.635895 | none |
| B | 91009 | `B_91009_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91010 | `B_91010_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.999964 | 2.635895 | none |
| B | 91011 | `B_91011_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91012 | `B_91012_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91013 | `B_91013_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91014 | `B_91014_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91015 | `B_91015_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91016 | `B_91016_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91017 | `B_91017_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |
| B | 91018 | `B_91018_contact_graph_feasibility_refine` | contact_graph_feasibility_refine | True | 0.983872 | 2.593476 | none |

### Skill Usage Summary

| Skill | Iteration uses |
|---|---:|
| archive-observability | 46 |
| evaluator-feedback | 46 |
| packing-repair | 42 |
| packing-slsqp | 2 |
| static-export | 2 |

### Strategy Archive Statistics

| Strategy | Attempts | Validity rate | Best score | Avg score improvement |
|---|---:|---:|---:|---:|
| benchmark_seed_dominikkamp | 2 | 1.000 | 0.999997 | 0.999997 |
| contact_graph_feasibility_refine | 34 | 1.000 | 0.999997 | -0.020925 |
| hexagonal_or_staggered_initialization | 2 | 1.000 | 0.915143 | -0.129478 |
| perturb_best_and_repair | 4 | 1.000 | 0.999997 | -0.000009 |
| public_frontier_dominikkamp | 2 | 1.000 | 0.999997 | 0.000000 |
| scipy_slsqp_joint | 2 | 1.000 | 0.985362 | -0.015380 |

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
  Elapsed : 0.17s
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
