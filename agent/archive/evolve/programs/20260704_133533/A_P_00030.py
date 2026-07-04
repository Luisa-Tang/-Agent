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
    threshold = float(context.get("contact_threshold", 5e-8) or 5e-8)
    scale = float(context.get("contact_scale", 0.35) or 0.35)
    midpoint = np.array([[0.5 * width, 0.5 * height]])
    direction = centers - midpoint
    norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    centers += direction / norm * rng.normal(0.0, threshold * scale, size=(len(centers), 1))
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "contact_threshold_mutation", "contact_threshold": threshold}}
# EVOLVE-BLOCK-END
