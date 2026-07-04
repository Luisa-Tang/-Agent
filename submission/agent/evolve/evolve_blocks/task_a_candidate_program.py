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
    sigma = float(context.get("sigma", 0.0) or 0.0)
    width_sigma = float(context.get("width_sigma", 0.0) or 0.0)
    if sigma > 0.0:
        centers += rng.normal(0.0, sigma, size=centers.shape)
    if width_sigma > 0.0:
        width += float(rng.normal(0.0, width_sigma))
    width = float(np.clip(width, 0.28, 1.72))
    height = 2.0 - width
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {
        "centers": centers,
        "radii": radii,
        "width": width,
        "height": height,
        "metadata": {
            "strategy_family": "baseline_parent_copy",
            "operator": context.get("operator", "seed"),
            "sigma": sigma,
            "width_sigma": width_sigma,
        },
    }
# EVOLVE-BLOCK-END
