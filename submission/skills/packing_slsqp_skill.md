# Packing SLSQP Skill

Purpose: generate high-scoring candidate centers and radii using deterministic SLSQP starts.

Use when:
- a valid archive seed exists but score is below the target band;
- structured layouts leave visible unused space;
- the manager selects `scipy_slsqp_joint` or `multi_start_slsqp`.

Procedure:
1. Build structured or jittered initial centers for the task geometry.
2. Optimize center coordinates and radii with SLSQP under boundary and pairwise constraints.
3. For Task A, include width as an optimized variable and set `height = 2 - width`.
4. Recompute radii for the final centers with the fixed-center LP repair step.
5. Export only if local validation and official evaluator validation pass.

Safety rules:
- Keep deterministic seeds in the candidate metadata.
- Preserve a positive safety margin before evaluator submission.
- Never modify official evaluator logic.
