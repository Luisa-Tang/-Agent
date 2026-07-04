# GEOMETRY Block

Allowed edits:
- transform centers;
- create structured perturbations;
- intentionally change contact graph or boundary pattern;
- use boundary slides, pair relaxation, small-circle repositioning, or group-aware moves.

Forbidden edits:
- hardcode final answer arrays;
- call official evaluators;
- modify files;
- return non-finite centers.

Inputs:
- `centers`, `radii`, `width`, `height`, `parent`, `rng`, `context`.

Outputs:
- finite centers approximately inside the container.

Verification:
- radius, refinement, safety, internal geometry checks, novelty filtering, and official evaluation.

Red flags:
- ordinary isotropic jitter with no intended geometry change;
- no `changed_indices` metadata.
