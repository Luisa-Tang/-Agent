# GeoEvolve-lite Depth Refinement

Patch only the EVOLVE-BLOCK to address the latest failure mode or plateau.
Use the provided active contact graph, boundary pattern, and strategy statistics
to make a small targeted change to the generator or refinement operator.

Prefer:
- preserving near-active contacts when the parent is already valid;
- adding controlled center perturbations when the archive is plateaued;
- changing a solver/margin parameter when official validity is fragile;
- emitting clear metadata that explains the attempted refinement.

The final submitted solution remains static and is exported only after official
evaluation succeeds.
