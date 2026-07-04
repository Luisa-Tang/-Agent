# Crossover Blocks

You may only combine named EVOLVE-BLOCK sections from parent programs.

Target:
Create a child program by taking geometry from one parent, radius solving from
another, and refinement/safety from a stable valid program.

Rules:
- Do not splice final coordinate arrays as the primary method.
- Do not hardcode final answers.
- Do not modify official evaluators.
- Preserve `propose_candidate(parent, rng, context)`.
- Preserve metadata and block usage tracking.
