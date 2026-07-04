---
name: packing-slsqp
description: Use this skill when generating or improving SciPy/SLSQP-based candidates for Task A or Task B circle-packing optimization. It guides the agent to let numeric optimizers handle coordinates and radii while the LLM controls parameterization, initialization, and search strategy.
---

# Packing SLSQP Skill

## Overview

This skill builds circle-packing candidates through constrained numeric optimization. The LLM should not manually guess all coordinates or radii. The LLM proposes parameterization, initialization families, margins, and optimization schedules; SciPy optimizes the continuous variables.

## When to Use

Use this skill when:

- Creating a new `solution.py` candidate for Task A or Task B.
- Improving a valid but low-scoring candidate.
- Adding a new initialization family such as grid, staggered, hex-like, boundary-aware, or warm-start layouts.
- Switching from naive construction to constrained local optimization.
- Improving the numeric quality of an existing candidate before static export.

Do not use this skill when:

- The candidate is invalid due to a simple interface error.
- The final `solution.py` only needs static array export.
- The evaluator failure is caused by missing imports, wrong return format, or wrong task directory.

## Inputs

Required inputs:

- Task name: `A` or `B`.
- Number of circles: Task A uses 21, Task B uses 26.
- Official evaluator output if available.
- Current best candidate from the archive if available.
- Selected initialization strategy.
- Safety margin, seed, optimizer method, and max iterations.

Useful files:

- `task_A/task_description.md`
- `task_B/task_description.md`
- `task_A/evaluate.py`
- `task_B/evaluate.py`
- `agent_runs/*/archive.jsonl`
- `agent/candidate_generators.py`

## Process

1. Read the task specification and official evaluator constraints.
2. Select the task-specific variable layout.
3. Select more than one initialization family: `safe_grid`, `staggered_rows`, `hex_like`, `boundary_aware`, `perturb_best`, or Task A `aspect_ratio_sweep`.
4. Generate initial centers and radii.
5. Build the SLSQP objective and constraints.
6. Run `scipy.optimize.minimize` with deterministic seed and bounded iterations.
7. Convert optimizer output to centers/radii/width/height.
8. Apply safety repair: clip only when appropriate, shrink radii near tolerance, and recompute `sum_radii` from radii.
9. Write a candidate `solution.py`.
10. Run the official evaluator.
11. Log optimizer method, seed, initialization family, maxiter, margin, validity, score, sum radii, and geometry margins.

## Decision Rules

- If no valid candidate exists, prefer `safe_grid` or conservative `staggered_rows`.
- If a valid candidate exists but score is low, try `multi_start_slsqp`.
- If SLSQP frequently fails, vary initialization and reduce initial radii.
- If the solution is valid but has very small margins, send it to `packing-repair`.
- If repeated attempts plateau, switch initialization family rather than only increasing iterations.
- For Task A, if score plateaus, run aspect-ratio sweep before perturbing individual coordinates.

## Task B Formulation

Optimize `x_i`, `y_i`, and `r_i`; maximize `sum(r_i)` by minimizing `-np.sum(radii)`. Bounds are `x_i,y_i in [0,1]` and `r_i in [0,0.5]`. Constraints are boundary margins and pairwise `dist(i,j) - r_i - r_j >= margin`.

## Task A Formulation

Optimize `x_i`, `y_i`, `r_i`, and `width`; compute `height = 2.0 - width`. Enforce positive width/height, perimeter through `height`, boundary margins, and pairwise non-overlap margins.

## Red Flags

- The LLM manually writes all coordinates without optimization or provenance.
- The generated code modifies `evaluate.py`.
- The candidate only passes a custom internal checker but not the official evaluator.
- The final `solution.py` runs expensive optimization every time it is imported.
- `sum_radii` is manually set and does not equal `np.sum(radii)`.
- Width and height do not satisfy `2 * (width + height) == 4`.

## Verification

A candidate produced by this skill is accepted only if the official evaluator passes, values are finite, shapes match, radii are non-negative, margins are non-negative, and the candidate plus evaluator output are saved in the archive.
