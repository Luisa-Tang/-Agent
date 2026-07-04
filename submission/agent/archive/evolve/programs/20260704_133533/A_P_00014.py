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
    mate = context.get("mate") or {}
    if "centers" in mate:
        mate_centers = np.asarray(mate["centers"], dtype=float)
        if mate_centers.shape == centers.shape:
            mask = (np.arange(len(centers)) % 2) == 1
            centers[mask] = 0.7 * centers[mask] + 0.3 * mate_centers[mask]
    if mate.get("width") is not None:
        width = float(np.clip(0.7 * width + 0.3 * float(mate.get("width")), 0.28, 1.72))
    height = 2.0 - width
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "crossover", "mate_available": bool(mate)}}
# EVOLVE-BLOCK-END
