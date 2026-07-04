---
name: evaluator-feedback
description: Use this skill after running the official evaluator. It maps evaluator output, score trajectory, and archive state into the next search action: repair, exploit, explore, restart, export, or stop.
---

# Evaluator Feedback Skill

## Overview

This skill turns official evaluator feedback into structured decisions. The evaluator is the only source of truth for validity. The Agent must not rely only on internal geometry checks or optimistic assumptions.

## When to Use

Use this skill after every candidate evaluation once `solution.py` has been written, the official evaluator has run, and stdout/stderr/return code have been captured. Also use it when score plateaus, validity rate drops, a strategy repeatedly fails, the archive must choose exploit/explore, or the Agent needs to decide whether to export.

## Inputs

Required inputs:

- Task name, candidate ID, parent candidate ID, and strategy name.
- Evaluator command, return code, stdout, stderr, parsed score, and parsed sum of radii.
- Current best score, current best valid candidate, and recent strategy statistics.

Optional inputs:

- Min pairwise margin, min boundary margin, optimizer status, elapsed time, candidate code hash, and prompt hash.

## Process

1. Classify failures deterministically before any LLM reasoning.
2. Map evaluator messages to `shape_error`, `nonfinite`, `negative_radius`, `perimeter_error`, `boundary_violation`, `overlap`, `timeout`, `low_score`, `plateau`, or `unknown`.
3. Choose an allowed next action: `repair`, `exploit`, `explore`, `restart`, `increase_budget`, `decrease_budget`, `switch_optimizer`, `static_export`, or `stop`.
4. Record `decision_reason` and `next_strategy`.
5. Preserve raw evaluator output and code snapshot paths.

## Decision Rules

- If invalid due to interface error, choose `repair` outside numeric optimization.
- If invalid due to geometry, choose `repair` when score potential is high, otherwise explore.
- If valid and improves best, archive as elite and exploit with perturb-and-repair.
- If valid but does not improve, archive as non-elite and explore if strategy is saturated.
- If score plateaus, switch initialization family or optimizer; do not only increase SLSQP iterations.
- If time budget is nearly exhausted, choose `static_export`.

## Red Flags

- The Agent ignores official evaluator failures.
- The Agent claims improvement based only on internal checks.
- The Agent repeats the same failed strategy many times without changing parameters.
- The Agent overwrites a best valid solution with an invalid candidate.
- The Agent stops without exporting the best valid candidate.

## Verification

This skill succeeds only if every evaluated candidate has a structured feedback record, every failure has a classified `failure_type`, every next action has a `decision_reason`, the best valid candidate is never lost, and the final report can reconstruct the iteration trajectory.
