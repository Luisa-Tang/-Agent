"""Default Task A candidate-generating program for GeoEvolve-lite.

The self-evolution harness may only replace code inside the EVOLVE-BLOCK.
This module is not a submitted solution; it proposes geometry for the harness,
which then runs repair, novelty filtering, and the official evaluator.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    height = 2.0 - width
    target_delta = float(context.get("target_delta", 1e-8) or 1e-8)
    midpoint = np.array([[0.5 * width, 0.5 * height]])
    direction = centers - midpoint
    centers += direction * target_delta * 0.08
    width = float(np.clip(width + target_delta * float(context.get("width_direction", 1.0)), 0.28, 1.72))
    height = 2.0 - width
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "depth_refinement", "target_delta": target_delta}}
# EVOLVE-BLOCK-END
