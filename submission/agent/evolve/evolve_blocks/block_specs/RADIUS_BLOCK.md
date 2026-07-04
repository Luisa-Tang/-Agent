# RADIUS Block

Allowed edits:
- call `context["solve_radius_lp"]`;
- choose parent radii with a safe shrink;
- set radius-solver metadata and parameters.

Forbidden edits:
- return inconsistent reported sums;
- bypass later safety checks;
- use external files or official evaluators.

Inputs:
- transformed centers and Task container.

Outputs:
- finite non-negative radii.

Verification:
- internal geometry checks, official evaluator, and safety guard.

Red flags:
- using stale parent radii after a significant geometry transform.
