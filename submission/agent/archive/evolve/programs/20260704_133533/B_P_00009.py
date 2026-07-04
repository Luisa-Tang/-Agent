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
    sigma = float(context.get("sigma", 3e-7) or 3e-7)
    centers += rng.normal(0.0, sigma, size=centers.shape)
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "parameter_mutation", "sigma": sigma}}
# EVOLVE-BLOCK-END
