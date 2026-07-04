# SAFETY Block

Allowed edits:
- clip centers to container;
- apply a tiny shrink;
- ensure non-negative finite radii;
- record safety metadata.

Forbidden edits:
- increase radii without a solver;
- hide validity failures;
- add network or file dependencies.

Inputs:
- current centers, radii, container.

Outputs:
- finite arrays suitable for cascade evaluation.

Verification:
- official evaluator is the final source of truth.

Red flags:
- excessive shrink that lowers score without improving validity.
