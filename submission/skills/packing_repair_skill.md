# Packing Repair Skill

Purpose: convert promising but risky geometry into robust evaluator-valid packing.

Use when:
- evaluator output reports overlap or boundary violation;
- SLSQP returns a candidate with near-zero safety margins;
- perturbing the best archive candidate requires radius recomputation.

Procedure:
1. Clip centers into the legal rectangle or unit square with a tiny epsilon.
2. Compute per-circle boundary limits.
3. Solve the fixed-center linear program maximizing `sum(radii)` under boundary and pairwise constraints.
4. Run pairwise shrink passes for any residual floating-point violation.
5. Record `min_pairwise_margin`, `min_boundary_margin`, `sum_radii`, and score.

Safety rules:
- Prefer slight shrinkage over invalid high scores.
- Keep all returned values finite and non-negative.
- Preserve Task B's `sum_radii == np.sum(radii)` invariant.
