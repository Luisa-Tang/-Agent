# Mutate Radius Solver Block

You may only edit code between:
`EVOLVE-BLOCK-RADIUS-START` and `EVOLVE-BLOCK-RADIUS-END`.

Target:
Choose a radius-solving policy that improves sum of radii for fixed centers.

Rules:
- Do not return stale radii after moving centers.
- Do not fake `sum_radii`.
- Do not call official evaluator.
- Update `metadata["blocks_used"]`.

Preferred ideas:
- LP with alternate margin;
- parent radii only with conservative shrink;
- radius group metadata for downstream refinement.
