# Mutate Refiner Block

You may only edit code between:
`EVOLVE-BLOCK-REFINE-START` and `EVOLVE-BLOCK-REFINE-END`.

Target:
Improve a candidate after radius LP while preserving official validity.

Rules:
- Keep all loops bounded.
- Do not call official evaluator.
- Do not write files or use network calls.
- Update `metadata["blocks_used"]`.

Preferred ideas:
- contact graph preserving correction;
- bounded coordinate descent;
- targeted repair for tight boundary or pair margins.
