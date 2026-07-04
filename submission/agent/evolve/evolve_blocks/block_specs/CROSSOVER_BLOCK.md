# CROSSOVER Block

Allowed edits:
- combine code blocks from multiple parent programs;
- take geometry from one parent and radius/refine/safety from another;
- preserve the `propose_candidate(parent, rng, context)` interface.

Forbidden edits:
- splice final coordinates from two parents as the primary crossover mechanism;
- hardcode final arrays;
- modify official evaluators or submitted solution files.

Inputs:
- parent program paths and optional mate payloads in context.

Outputs:
- a syntactically valid child program that still goes through cascade evaluation.

Verification:
- block hashes, program DB lineage, novelty filter, official evaluator.

Red flags:
- crossed blocks that omit required metadata or container clipping.
