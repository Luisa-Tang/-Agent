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
    sigma = float(context.get("sigma", 1e-6) or 1e-6)
    order = np.argsort(-radii)
    for rank, idx in enumerate(order[: max(1, len(order) // 3)]):
        angle = 2.399963229728653 * (rank + 1)
        centers[idx, 0] += sigma * np.cos(angle)
        centers[idx, 1] += sigma * np.sin(angle)
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "program_patch_fallback", "llm_used": False, "sigma": sigma}}
# EVOLVE-BLOCK-END
