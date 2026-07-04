# Mutate Geometry Block

You may only edit code between:
`EVOLVE-BLOCK-GEOMETRY-START` and `EVOLVE-BLOCK-GEOMETRY-END`.

Target:
Create a geometry transform that is more likely to change the active contact
graph or boundary pattern than ordinary Gaussian jitter.

Use context:
- parent contact graph and boundary pattern;
- tight circle pairs;
- smallest circles;
- recent failed transforms;
- operator statistics.

Rules:
- Do not call official evaluator.
- Do not hardcode final arrays.
- Do not modify radius, refinement, or safety blocks.
- Return finite centers and update metadata.

Preferred ideas:
- boundary slide;
- small-circle reposition;
- contact pair relaxation;
- aspect-ratio-aware scaling for Task A.
