"""Default Task B candidate-generating program for GeoEvolve-lite.

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
    mate = context.get("mate") or {}
    if "centers" in mate:
        mate_centers = np.asarray(mate["centers"], dtype=float)
        if mate_centers.shape == centers.shape:
            mask = (np.arange(len(centers)) % 2) == 1
            centers[mask] = 0.7 * centers[mask] + 0.3 * mate_centers[mask]
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "crossover", "mate_available": bool(mate)}}
# EVOLVE-BLOCK-END
