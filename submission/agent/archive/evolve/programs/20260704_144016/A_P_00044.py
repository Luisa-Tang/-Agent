"""Task A candidate-generating program for GeoEvolve-lite.

The outer EVOLVE-BLOCK keeps v1 compatibility. Evolve Blocks v2 mutates only
the named inner blocks so aspect ratio, geometry, radius, refinement, and
safety can be tracked independently.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    height = 2.0 - width
    metadata = {
        "task": "A",
        "blocks_used": [],
        "strategy_family": "baseline_parent_copy",
        "operator_name": context.get("operator", "seed"),
    }

    # EVOLVE-BLOCK-ASPECT-START
    old_width = float(width)
    old_height = float(height)
    deltas = context.get("aspect_bucket_deltas") or [1e-5, 3e-5, 1e-4, 3e-4]
    delta = float(deltas[int(context.get("generation", 0) or 0) % len(deltas)])
    direction = float(context.get("width_direction", 1.0) or 1.0)
    width = float(np.clip(width + direction * delta, 0.28, 1.72))
    height = 2.0 - width
    centers[:, 0] *= width / max(old_width, 1e-12)
    centers[:, 1] *= height / max(old_height, 1e-12)
    old_bucket = int(np.floor(old_width / 3e-5))
    new_bucket = int(np.floor(width / 3e-5))
    metadata["blocks_used"].append("aspect:aspect_ratio_island")
    metadata["operator_name"] = "aspect_ratio_island"
    metadata["old_width"] = old_width
    metadata["new_width"] = width
    metadata["width_delta"] = width - old_width
    metadata["aspect_bucket"] = new_bucket
    metadata["aspect_ratio_bucket_changed"] = bool(old_bucket != new_bucket)
    metadata["parameters"] = {"aspect_delta": delta, "width_direction": direction}
    # EVOLVE-BLOCK-ASPECT-END

    # EVOLVE-BLOCK-GEOMETRY-START
    mode = "aspect_ratio_island"
    k = int(context.get("destroy_k", 2) or 2)
    if mode == "small_circle_swap":
        k = max(2, min(k, 3))
    elif mode == "boundary_gap_refill":
        k = max(1, min(k, 2))
    else:
        k = max(2, min(k, 4))
    gap_helper = context.get("gap_refill")
    if callable(gap_helper):
        centers, gap_meta = gap_helper(centers=centers, radii=radii, width=width, height=height, k=k, mode=mode)
    else:
        order = np.argsort(radii)[:k]
        gap_meta = {"removed_indices": [int(i) for i in order], "gap_ids": [], "small_circle_reassigned": True}
        for idx in order:
            centers[idx] += rng.normal(0.0, float(context.get("gap_probe_scale", 0.035) or 0.035), size=2)
    changed_indices = [int(i) for i in gap_meta.get("removed_indices", [])]
    metadata["blocks_used"].append("geometry:aspect_ratio_island")
    metadata["strategy_family"] = "aspect_ratio_island"
    metadata["operator_name"] = "aspect_ratio_island"
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
        radii = solver(centers=centers, container=(width, height), task="A")
    metadata["blocks_used"].append("radius:lp")
    # EVOLVE-BLOCK-RADIUS-END

    # EVOLVE-BLOCK-REFINE-START
    refiner = context.get("repair_and_polish_task_a")
    if callable(refiner) and context.get("enable_fast_refine", False):
        centers, radii, width, height = refiner(
            centers=centers,
            radii=radii,
            width=width,
            height=height,
            max_steps=int(context.get("max_refine_steps", 8) or 8),
        )
        metadata["blocks_used"].append("refine:task_a_repair_polish")
    else:
        metadata["blocks_used"].append("refine:none")
    # EVOLVE-BLOCK-REFINE-END

    # EVOLVE-BLOCK-SAFETY-START
    width = float(np.clip(width, 0.28, 1.72))
    height = 2.0 - width
    centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)
    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)
    radii = np.maximum(np.asarray(radii, dtype=float), 0.0)
    shrink = 1.0 - float(context.get("safety_shrink", 1e-10) or 1e-10)
    radii *= shrink
    metadata["blocks_used"].append("safety:perimeter_shrink")
    # EVOLVE-BLOCK-SAFETY-END

    return {
        "centers": centers,
        "radii": radii,
        "width": width,
        "height": height,
        "metadata": metadata,
    }
# EVOLVE-BLOCK-END
