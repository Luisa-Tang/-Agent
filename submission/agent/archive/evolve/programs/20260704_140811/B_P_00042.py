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
    count = max(1, int(context.get("small_circle_count", 2) or 2))
    order = np.argsort(radii)[:count]
    box = np.array([1.0, 1.0], dtype=float)
    changed_indices = []
    anchors = centers[np.argsort(-radii)[: max(1, min(5, len(radii)))]]
    for rank, idx in enumerate(order):
        if len(anchors):
            anchor = anchors[int(rng.integers(0, len(anchors)))]
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            dist = float(context.get("gap_probe_scale", 0.035) or 0.035) * min(box)
            centers[idx] = anchor + dist * np.array([np.cos(angle), np.sin(angle)])
        else:
            centers[idx] = rng.uniform(0.05, 0.95, size=2) * box
        changed_indices.append(int(idx))
    centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)
    centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)
    metadata["blocks_used"].append("geometry:small_circle_reposition")
    metadata["strategy_family"] = "small_circle_reposition"
    metadata["operator_name"] = "small_circle_reposition"
    metadata["intended_contact_change"] = "move smallest circles into nearby gaps"
    metadata["intended_boundary_change"] = "possible"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"small_circle_count": count}
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
