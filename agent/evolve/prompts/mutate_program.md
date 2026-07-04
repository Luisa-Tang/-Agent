# GeoEvolve-lite Program Mutation

You are editing a candidate-generating program for a circle-packing search
harness. Modify only the code between `# EVOLVE-BLOCK-START` and
`# EVOLVE-BLOCK-END`.

Hard rules:
- Do not modify official evaluators, task descriptions, or final solution files.
- Do not write static final coordinates by hand.
- The function must keep the signature `propose_candidate(parent, rng, context)`.
- The function must return a dict containing `centers`, `radii`, optional
  Task A `width` and `height`, and `metadata`.
- The output is only a candidate; the official evaluator decides validity.

Goal:
Improve the candidate-generating logic while preserving reproducibility and
offline execution.
