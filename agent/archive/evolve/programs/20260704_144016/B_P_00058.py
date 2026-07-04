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
    mode = "boundary_gap_refill"
    k = int(context.get("destroy_k", 2) or 2)
    if mode == "small_circle_swap":
        k = max(2, min(k, 3))
    elif mode == "boundary_gap_refill":
        k = max(1, min(k, 2))
    else:
        k = max(2, min(k, 4))
    gap_helper = context.get("gap_refill")
    if callable(gap_helper):
        centers, gap_meta = gap_helper(centers=centers, radii=radii, width=1.0, height=1.0, k=k, mode=mode)
    else:
        order = np.argsort(radii)[:k]
        gap_meta = {"removed_indices": [int(i) for i in order], "gap_ids": [], "small_circle_reassigned": True}
        for idx in order:
            centers[idx] += rng.normal(0.0, float(context.get("gap_probe_scale", 0.035) or 0.035), size=2)
    changed_indices = [int(i) for i in gap_meta.get("removed_indices", [])]
    metadata["blocks_used"].append("geometry:boundary_gap_refill")
    metadata["strategy_family"] = "boundary_gap_refill"
    metadata["operator_name"] = "boundary_gap_refill"
    metadata["intended_contact_change"] = "destroy small-circle placement and refill high-scoring gaps"
    metadata["intended_boundary_change"] = "possible boundary gap refill"
    metadata["changed_indices"] = changed_indices
    metadata["removed_indices"] = changed_indices
    metadata["inserted_gap_ids"] = list(gap_meta.get("gap_ids", []))
    metadata["gap_sources"] = list(gap_meta.get("gap_sources", []))
    metadata["small_circle_reassigned"] = bool(gap_meta.get("small_circle_reassigned", False))
    metadata["top_gap_score"] = float(gap_meta.get("top_gap_score", 0.0) or 0.0)
    metadata["top_gap_radius"] = float(gap_meta.get("top_gap_radius", 0.0) or 0.0)
    metadata["parameters"] = {"destroy_k": k, "mode": mode}
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
