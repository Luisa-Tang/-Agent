---
name: archive-observability
description: Use this skill when recording candidate lineage, evaluator outputs, strategy statistics, safety metrics, and report-ready traces for the circle-packing optimization agent.
---

# Archive Observability Skill

## Overview

This skill ensures that the Agent's search process is traceable, reproducible, and reportable. The archive should not only store the best score. It must explain why candidates improved, failed, were repaired, or were rejected.

## When to Use

Use this skill at every iteration, after every official evaluator run, after repair attempts, when a candidate becomes best-so-far, when exporting final solutions, and when generating the final report. Observability must be built into the loop, not added only at the end.

## Inputs

Required inputs:

- Candidate metadata, parent ID, strategy, generator, optimizer, seed, validity, score, sum radii, width/height, failure type, repair action, elapsed time, raw output path, code snapshot path, and geometry safety metrics.

Optional inputs:

- Prompt hash, code hash, aspect ratio, perimeter error, Task B returned-sum error, and strategy-level aggregate state.

## Process

1. Save every candidate code snapshot.
2. Save every evaluator stdout/stderr blob.
3. Append one JSON object per iteration to archive JSONL.
4. Record observe -> think -> act -> observe trace.
5. Maintain per-strategy attempts, validity rate, best score, and score improvement.
6. Preserve parent-child lineage and source candidate IDs for exports.
7. Generate report-ready tables for trajectory, failures, strategy comparison, and final metrics.

## Decision Rules

- Never discard invalid candidates without a failure record.
- Never call a candidate best without official evaluator evidence.
- If a candidate improves best score, record its lineage and selected strategy.
- If a strategy plateaus or fails repeatedly, expose that in strategy statistics.
- If final export occurs, record source candidate ID and post-export evaluator output.

## Red Flags

- Only `best_score` is logged.
- Invalid candidates are discarded without failure records.
- Raw evaluator outputs are missing.
- Parent-child lineage is missing.
- Strategy-level statistics do not exist.
- The final report cannot reconstruct how the final solution was found.

## Verification

This skill succeeds only if archive JSONL exists, every candidate has a code snapshot, every evaluator run has raw output saved, best valid candidates trace to candidate IDs, final exported solutions reference source candidate IDs, and the report includes best score, trajectory, failure-type summary, strategy comparison, and human-agent division.
