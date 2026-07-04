# AGENTS.md

## Project role

You are working on an AlgorithmOptimization competition project. The goal is not only to produce high-scoring circle-packing solutions, but also to implement a reproducible local Agent system that can read task descriptions, generate candidate solution.py files, run the official evaluators, parse feedback, iterate, log decisions, and export the final submission package.

## Hard rules

* Do not modify:

  * evaluate_all.py
  * task_A/evaluate.py
  * task_B/evaluate.py
  * task_A/task_description.md
  * task_B/task_description.md
  * any official evaluation logic

* You may read baseline.py and task_description.md.
* The final solution files must strictly satisfy the required interfaces:

  * task_A/solution.py must define run_packing(num_circles) and return centers, radii, width, height.
  * task_B/solution.py must define run_packing(num_circles) and return centers, radii, sum_radii.

* All outputs must be finite numeric values.
* All circles must be inside the required rectangle or unit square.
* No pair of circles may overlap.
* The generated solution.py files should be deterministic or nearly deterministic unless the random seed is explicitly fixed.
* Prefer robust legal packings over risky high-scoring packings that may fail due to floating-point tolerance.
* Do not depend on unavailable services for final evaluation. The final solution.py files must run without LLM API access.

## Engineering expectations

* Use Python 3.
* Prefer numpy and scipy. Add a requirements.txt only if necessary.
* Keep the agent implementation simple and reproducible.
* Use subprocess to run the official evaluators instead of reimplementing their scoring logic as the only source of truth.
* Record every iteration in JSONL or JSON format.
* Store candidate solution files and best solution snapshots.
* Include enough logs for the report to explain the Agent's autonomous search behavior.

## Optimization strategy expectations

The Agent should combine several strategies rather than relying on a single prompt-generated solution:

1. Template-based solver generation:

   * SLSQP-based joint optimization.
   * Multi-start random initialization.
   * Hexagonal / staggered / grid-like initialization.
   * Perturb-and-repair around best candidates.
   * Safety-margin shrinking to avoid overlaps and boundary failures.

2. Feedback-based repair:

   * If evaluator reports overlap, add stronger pairwise constraints or shrink radii.
   * If evaluator reports outside-boundary failure, add boundary constraints or safety margins.
   * If score is valid but low, explore new initialization families.
   * If scipy optimization fails, fall back to a previous valid candidate.

3. Archive and selection:

   * Keep all valid candidates.
   * Track best score, sum of radii, parameters, strategy, and parent candidate.
   * Export the best valid candidate into task_A/solution.py and task_B/solution.py.

## Deliverables

The final repository must contain:

```text
submission/
├── agent/
│   ├── run.py
│   ├── evaluator_adapter.py
│   ├── candidate_generators.py
│   ├── archive.py
│   ├── report_data.py
│   └── prompts/
├── task_A/
│   ├── solution.py
│   └── run_log_a.log
├── task_B/
│   ├── solution.py
│   └── run_log_b.log
└── report.md or report.pdf
```

If PDF generation is difficult in the environment, create a high-quality report.md and clearly document how to convert it to PDF.

## Validation commands

Before finishing, run:

```bash
python task_A/evaluate.py
python task_B/evaluate.py
python evaluate_all.py
```

Also run the Agent entrypoint at least once:

```bash
python agent/run.py --task both --iterations 5 --fast
```

The full run may support more iterations, but the fast run must complete quickly and prove reproducibility.
