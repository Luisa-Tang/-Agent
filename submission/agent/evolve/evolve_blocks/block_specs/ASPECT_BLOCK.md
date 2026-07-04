# ASPECT Block

Applies only to Task A.

Allowed edits:
- perturb `width` in a controlled way;
- choose width from an archive aspect-ratio bucket;
- keep `height = 2.0 - width`.

Forbidden edits:
- violate `width + height = 2`;
- call official evaluators;
- write files or hardcode final arrays.

Inputs:
- `parent`, `rng`, `context`, current `width`, `height`, `centers`, `radii`.

Outputs:
- finite `width` and `height` in a feasible range.

Verification:
- later blocks and the cascade evaluator clip centers, solve radii, and run official validation.

Red flags:
- large random width jumps near a tight benchmark solution;
- changes that do not record `metadata["blocks_used"]`.
