# AlgorithmOptimization Local Agent Report

## 1. Overview

This repository contains a deterministic local Agent system that generates `solution.py` candidates for two circle-packing optimization tasks, runs the official evaluators through subprocesses, parses feedback, archives every attempt, and exports the best valid candidates. The harness follows an observe -> think -> act -> observe loop so every code-evolution step is traceable.

## 2. System Architecture

- **ProblemParser**: `agent/run.py` reads task descriptions and evaluator files before the loop starts.
- **CandidateGenerator**: `agent/candidate_generators.py` creates standalone solution code from safe grids, staggered starts, SLSQP search, multi-start search, and perturb-and-repair.
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

## 3. Code Generation Strategy

The Agent uses template-based generation. Each candidate is a complete Python module with literal NumPy arrays for centers and radii, plus a small safety repair routine. Candidate search combines conservative grid layouts, hexagonal/staggered initializations, SLSQP joint optimization over centers and radii, multi-start SLSQP, and local perturbation around the best valid candidate. For fixed centers, the Agent solves a linear program to maximize radii under boundary and pairwise non-overlap constraints, then applies a tiny final shrink/repair.

LLM use is optional. When `--use-llm` is passed and `OPENAI_API_KEY` is set, the Agent asks a configured Chat Completions-compatible endpoint to choose among the existing deterministic strategies. The LLM is not allowed to modify official evaluators, and final exported `solution.py` files remain standalone with no network or LLM dependency.

## 4. Feedback Utilization

Evaluator output is parsed for score, `sum_radii`, validity, and failure type. Overlap failures trigger more conservative repair. Outside-boundary failures trigger boundary-tight generation. Low but valid scores move the Agent toward multi-start and structured initializations. Plateaued valid runs trigger perturb-and-repair around the current best candidate.

Failure classification uses the explicit taxonomy: `shape_error`, `nonfinite`, `negative_radius`, `perimeter_error`, `boundary_violation`, `overlap`, `timeout`, `low_score`, `plateau`, and `unknown`.

## 5. Termination and Decision Mechanism

The loop terminates after the configured iteration budget or time budget. The archive keeps every candidate, but only official-evaluator-valid candidates can become final exports. If no valid optimized candidate is available, the deterministic safe grid fallback remains valid.

## 6. Results

- Task A best candidate: `A_003_multi_start_slsqp`
- Task A sum_radii: `2.349608`
- Task A score: `0.993139` using denominator `2.365840`
- Task A width/height: `0.932160847` / `1.067839153`
- Task B best candidate: `B_003_multi_start_slsqp`
- Task B sum_radii: `2.607923`
- Task B score: `0.989352` using denominator `2.635990`
- Combined best-valid score: `0.991246`

### Iteration Trajectory

| Task | Iteration | Candidate | Strategy | Valid | Score | Sum radii | Failure |
|---|---:|---|---|---:|---:|---:|---|
| A | 0 | `A_000_baseline_safe_grid` | baseline_safe_grid | True | 0.887633 | 2.099998 | low_score |
| A | 1 | `A_001_hexagonal_or_staggered_initialization` | hexagonal_or_staggered_initialization | True | 0.825895 | 1.953936 | low_score |
| A | 2 | `A_002_scipy_slsqp_joint` | scipy_slsqp_joint | True | 0.990479 | 2.343314 | none |
| A | 3 | `A_003_multi_start_slsqp` | multi_start_slsqp | True | 0.993139 | 2.349608 | none |
| A | 4 | `A_004_perturb_best_and_repair` | perturb_best_and_repair | True | 0.993139 | 2.349608 | none |
| A | 5 | `A_005_hexagonal_or_staggered_initialization` | hexagonal_or_staggered_initialization | True | 0.743962 | 1.760095 | low_score |
| A | 6 | `A_006_perturb_best_and_repair` | perturb_best_and_repair | True | 0.993139 | 2.349608 | plateau |
| A | 7 | `A_007_perturb_best_and_repair` | perturb_best_and_repair | True | 0.993139 | 2.349608 | plateau |
| B | 0 | `B_000_baseline_safe_grid` | baseline_safe_grid | True | 0.821955 | 2.166665 | low_score |
| B | 1 | `B_001_hexagonal_or_staggered_initialization` | hexagonal_or_staggered_initialization | True | 0.915143 | 2.412309 | none |
| B | 2 | `B_002_scipy_slsqp_joint` | scipy_slsqp_joint | True | 0.979240 | 2.581266 | none |
| B | 3 | `B_003_multi_start_slsqp` | multi_start_slsqp | True | 0.989352 | 2.607923 | none |
| B | 4 | `B_004_perturb_best_and_repair` | perturb_best_and_repair | True | 0.989352 | 2.607923 | none |
| B | 5 | `B_005_hexagonal_or_staggered_initialization` | hexagonal_or_staggered_initialization | True | 0.915143 | 2.412309 | none |
| B | 6 | `B_006_perturb_best_and_repair` | perturb_best_and_repair | True | 0.989352 | 2.607923 | plateau |
| B | 7 | `B_007_perturb_best_and_repair` | perturb_best_and_repair | True | 0.989352 | 2.607923 | plateau |

### Skill Usage Summary

| Skill | Iteration uses |
|---|---:|
| archive-observability | 16 |
| evaluator-feedback | 16 |
| packing-repair | 12 |
| packing-slsqp | 4 |

### Strategy Archive Statistics

| Strategy | Attempts | Validity rate | Best score | Avg score improvement |
|---|---:|---:|---:|---:|
| baseline_safe_grid | 2 | 1.000 | 0.887633 | 0.854794 |
| hexagonal_or_staggered_initialization | 4 | 1.000 | 0.915143 | -0.072984 |
| multi_start_slsqp | 2 | 1.000 | 0.993139 | 0.006386 |
| perturb_best_and_repair | 6 | 1.000 | 0.993139 | 0.000000 |
| scipy_slsqp_joint | 2 | 1.000 | 0.990479 | 0.083471 |

### Best Geometry Safety Metrics

| Task | Min pairwise margin | Min boundary margin | Sum radii | Score | Width | Height |
|---|---:|---:|---:|---:|---:|---:|
| A | 7.995e-08 | 8.000e-08 | 2.349608 | 0.993139 | 0.932160847 | 1.067839153 |
| B | 8.000e-08 | 8.000e-08 | 2.607923 | 0.989352 |  |  |

### Final Evaluator Output

```text
============================================================
  Circle Packing in Rectangle  (n=21)
  File : /home/wuyou/projects/AlgorithmOptimization/task_A/solution.py
============================================================
  Elapsed : 0.18s
  sum_radii : 2.349608
  Target    : 2.365840
  Score     : 0.993139
============================================================


============================================================
  Circle Packing in Unit Square  (n=26)
  File : /home/wuyou/projects/AlgorithmOptimization/task_B/solution.py
============================================================
  Elapsed : 0.17s
  sum_radii : 2.607923
  Target    : 2.635990
  Score     : 0.989352
============================================================

============================================================
  Final Score
============================================================
  task_A (Circle Packing in Rectangle)   :  0.993139
  task_B (Circle Packing in Unit Square) :  0.989352
  Combined                               :  0.991246
============================================================
```

## 7. Human-Agent Division

The human provided the high-level system design, constraints, required interfaces, and quality bar. The Agent implemented the framework, ran the local evaluator loop, generated candidate solution files, selected the best valid candidates, exported artifacts, and produced logs. Optional LLM connectivity is supported but the recorded run may use deterministic fallback unless `--use-llm` and credentials are provided. The main manual assumption was using the local conda Python environment when the system `python` command was unavailable.

## 8. Limitations and Future Work

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
