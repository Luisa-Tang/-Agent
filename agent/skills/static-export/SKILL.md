---
name: static-export
description: Use this skill when exporting the best archived circle-packing candidate into the final task_A/solution.py or task_B/solution.py. It ensures deterministic, lightweight, evaluator-safe output.
---

# Static Export Skill

## Overview

This skill converts the best valid candidate from the search archive into a final deterministic `solution.py`. The final file should be standalone, fast, stable, and compatible with the official evaluator. Expensive optimization belongs in the Agent loop, not in final evaluation.

## When to Use

Use this skill when the archive contains at least one valid candidate, the Agent nears iteration/time budget, the final package is being created, a high-scoring optimized candidate must be frozen into arrays, or the evaluator times out because `solution.py` runs too much optimization.

Do not use this skill when no valid candidate exists, the candidate has not passed the official evaluator, or margins require repair first.

## Inputs

Required inputs:

- Best valid candidate ID and task name.
- Centers, radii, width/height for Task A, and sum of radii for Task B.
- Official evaluator output confirming validity.
- Geometry safety metrics.

Optional inputs:

- Code snapshot, repair history, and parent lineage.

## Process

1. Select the best valid candidate by official evaluator score.
2. Copy or render static NumPy arrays into the task `solution.py`.
3. Include only necessary imports and lightweight safety hardening.
4. Recompute Task B `sum_radii` from radii.
5. Re-run official task evaluator and `evaluate_all.py solution.py`.
6. Copy final solutions, logs, archive summary, skills, and report into `submission/`.

## Decision Rules

- Use safety shrink `1.0 - 1e-12` for comfortable margins, `1.0 - 1e-10` for small margins, and `1.0 - 1e-8` only for fragile evaluator tolerance.
- Do not shrink enough to materially reduce score unless necessary for validity.
- For Task A compute `height = 2.0 - width` or verify exact perimeter consistency.

## Red Flags

- Final `solution.py` calls an LLM API, downloads data, modifies files, changes official evaluator, or performs expensive optimization on every evaluation.
- Task B returns a hardcoded sum inconsistent with `np.sum(radii)`.
- Task A uses independent width and height values that may violate perimeter due to rounding.

## Verification

After export, run the official evaluators and `evaluate_all.py`. The export is accepted only if official evaluator passes, final files are standalone and deterministic, runtime is acceptable, archive records exported candidate ID, and report includes final score and export procedure.
