"""Task B candidate-generating program for GeoEvolve-lite.

The outer EVOLVE-BLOCK keeps v1 compatibility. Evolve Blocks v2 mutates only
the named inner blocks so geometry, radius, refinement, and safety can be
tracked independently.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    metadata = {
        "task": "B",
        "blocks_used": [],
        "strategy_family": "baseline_parent_copy",
        "operator_name": context.get("operator", "seed"),
    }

    # EVOLVE-BLOCK-GEOMETRY-START
    edge_margin = np.minimum.reduce([centers[:, 0], 1.0 - centers[:, 0], centers[:, 1], 1.0 - centers[:, 1]])
    band = float(context.get("edge_band", 5e-7) or 5e-7)
    step = float(context.get("sigma", 3e-7) or 3e-7)
    idx = np.where(edge_margin <= np.percentile(edge_margin, 35))[0]
    changed_indices = idx.astype(int).tolist()
    for k in idx:
        left = centers[k, 0]
        right = (width - centers[k, 0]) if metadata.get("task") == "A" else (1.0 - centers[k, 0])
        bottom = centers[k, 1]
        top = (height - centers[k, 1]) if metadata.get("task") == "A" else (1.0 - centers[k, 1])
        side = int(np.argmin([left, right, bottom, top]))
        if side in (0, 1):
            centers[k, 1] += rng.normal(0.0, step)
        else:
            centers[k, 0] += rng.normal(0.0, step)
    centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)
    centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)
    metadata["blocks_used"].append("geometry:boundary_slide")
    metadata["strategy_family"] = "boundary_slide_mutation"
    metadata["operator_name"] = "boundary_slide_mutation"
    metadata["intended_contact_change"] = "preserve local boundary contacts while changing tangential order"
    metadata["intended_boundary_change"] = "slide active boundary circles"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"edge_band": band, "step": step}
    # EVOLVE-BLOCK-GEOMETRY-END

    # EVOLVE-BLOCK-RADIUS-START
    solver = context.get("solve_radius_lp")
    if callable(solver):
        radii = solver(centers=centers, container=(1.0, 1.0), task="B")
    metadata["blocks_used"].append("radius:lp")
    # EVOLVE-BLOCK-RADIUS-END

    # EVOLVE-BLOCK-REFINE-START
    refiner = context.get("repair_and_polish")
    if callable(refiner) and context.get("enable_fast_refine", False):
        centers, radii = refiner(
            centers=centers,
            radii=radii,
            task="B",
            max_steps=int(context.get("max_refine_steps", 8) or 8),
        )
        metadata["blocks_used"].append("refine:repair_polish")
    else:
        metadata["blocks_used"].append("refine:none")
    # EVOLVE-BLOCK-REFINE-END

    # EVOLVE-BLOCK-SAFETY-START
    radii = np.maximum(np.asarray(radii, dtype=float), 0.0)
    shrink = 1.0 - float(context.get("safety_shrink", 1e-10) or 1e-10)
    radii *= shrink
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    metadata["blocks_used"].append("safety:shrink")
    # EVOLVE-BLOCK-SAFETY-END

    return {
        "centers": centers,
        "radii": radii,
        "metadata": metadata,
    }
# EVOLVE-BLOCK-END
