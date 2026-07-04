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
    width_sigma = float(context.get("width_sigma", 1e-7) or 1e-7)
    if width_sigma > 0.0:
        width += float(rng.normal(0.0, width_sigma))
    width = float(np.clip(width, 0.28, 1.72))
    height = 2.0 - width
    metadata["blocks_used"].append("aspect:micro")
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
        radii = solver(centers=centers, container=(width, height) if metadata.get("task") == "A" else (1.0, 1.0), task=metadata.get("task"))
    else:
        radii = np.maximum(radii, 0.0)
    metadata["blocks_used"].append("radius:solver_switch_lp")
    metadata["operator_name"] = "solver_switch"
    metadata["parameters"] = {"solver": context.get("solver", "lp")}
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
