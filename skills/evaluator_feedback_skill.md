# Evaluator Feedback Skill

Purpose: map official evaluator output into structured feedback for the manager loop.

Failure taxonomy:
- `shape_error`
- `nonfinite`
- `negative_radius`
- `perimeter_error`
- `boundary_violation`
- `overlap`
- `timeout`
- `low_score`
- `plateau`
- `unknown`

Procedure:
1. Run the official evaluator through `EvaluatorAdapter`; do not reimplement scoring as authority.
2. Parse `Score`, `sum_radii`, elapsed time, and failure text from stdout/stderr.
3. Classify invalid output from evaluator messages.
4. Classify valid but weak output as `low_score` when below the local score band.
5. Classify repeated no-improvement valid output as `plateau`.
6. Store the raw output path and compact observation in the archive JSONL.

Safety rules:
- Raw evaluator output must be retained for auditability.
- Final export must come from a candidate with official evaluator validity.
