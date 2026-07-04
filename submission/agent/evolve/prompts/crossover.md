# GeoEvolve-lite Crossover

Patch only the EVOLVE-BLOCK to combine two parent strategies or geometries.
The second parent may be available as `context["mate"]` with `centers`,
`radii`, and optional Task A `width` / `height`.

Hard rules:
- keep the `propose_candidate(parent, rng, context)` signature;
- return arrays with the required task shape;
- do not call the network or external programs;
- do not edit official evaluators.
