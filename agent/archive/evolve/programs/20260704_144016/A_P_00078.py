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
    n = len(radii)
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                pairs.append((d - float(radii[i] + radii[j]), i, j))
        pairs.sort(key=lambda item: item[0])
        _m, i, j = pairs[int(context.get("pair_rank", 0) or 0) % len(pairs)]
        direction = centers[i] - centers[j]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction = direction / norm
            eps = float(context.get("target_delta", 1e-8) or 1e-8)
            centers[i] += eps * direction
            centers[j] -= eps * direction
            changed_indices = [int(i), int(j)]
        else:
            changed_indices = []
    else:
        changed_indices = []
    centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)
    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)
    metadata["blocks_used"].append("geometry:contact_graph_breaking_refine")
    metadata["strategy_family"] = "contact_graph_breaking_refine"
    metadata["operator_name"] = "contact_graph_breaking_refine"
    metadata["intended_contact_change"] = "break one active edge to search a new basin"
    metadata["intended_boundary_change"] = "none"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"target_delta": float(context.get("target_delta", 1e-8) or 1e-8)}
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
