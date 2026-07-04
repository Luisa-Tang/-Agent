---
name: packing-repair
description: Use this skill when a circle-packing candidate is invalid, numerically unsafe, overlapping, outside the container, or too close to evaluator tolerance. It defines deterministic repair actions before asking the LLM to rewrite code.
---

# Packing Repair Skill

## Overview

This skill repairs invalid or unsafe circle-packing candidates. It should be used before discarding a high-scoring candidate. The purpose is to convert near-valid candidates into robust valid candidates while preserving as much score as possible.

## When to Use

Use this skill when the evaluator reports outside-boundary circles, overlap, non-finite values, negative radii, or Task A perimeter errors. Also use it when internal metrics show negative or fragile `min_pairwise_margin` or `min_boundary_margin`.

Do not use this skill when the function interface is missing, generated code has syntax errors, the candidate is obviously low-scoring and not worth repairing, or required dependencies are unavailable.

## Inputs

Required inputs:

- Candidate centers and radii.
- Task name.
- Width and height for Task A.
- Official evaluator error message.
- Internal geometry metrics if available.

Optional inputs:

- Parent candidate ID.
- Strategy name that produced the candidate.
- Safety target such as `1e-7` or `1e-6`.

## Process

1. Classify the failure as interface, nonfinite, negative_radius, perimeter, boundary, overlap, unsafe_margin, timeout, or unknown.
2. Compute minimum pairwise and boundary margins.
3. Apply deterministic repair: recover from parent for non-finite values, clamp tiny negative radii, enforce Task A `height = 2.0 - width`, shrink affected radii for boundary violations, shrink one or both radii for overlaps, and apply global shrink when many overlaps exist.
4. If the candidate is near-valid, call SLSQP polish with the repaired candidate as initialization.
5. Re-run the official evaluator.
6. Accept only if the official evaluator passes.

## Decision Rules

- If repair score loss is less than `1e-5` relative, accept repaired candidate.
- If repair score loss is larger but the original candidate was invalid, accept only if it becomes the best valid candidate.
- If repair repeatedly destroys score, return control to `packing-slsqp` with stronger constraints.
- If Task A boundary violations concentrate on one side, trigger aspect-ratio adjustment rather than only shrinking radii.

## Red Flags

- Repair hides a modeling bug by shrinking all radii to near zero.
- Repair modifies official evaluator files.
- Repair accepts a candidate without re-running official evaluator.
- Repair returns `sum_radii` inconsistent with `np.sum(radii)`.
- Repair changes width/height without preserving Task A perimeter.

## Verification

The repaired candidate is valid only if the official evaluator passes, margins are non-negative, score and sum of radii are recomputed after repair, and the archive records original candidate ID, repair action, failure type, before/after score, and evaluator output path.
