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
    q1, q2 = np.quantile(radii, [0.33, 0.67])
    small = np.where(radii <= q1)[0]
    medium = np.where((radii > q1) & (radii <= q2))[0]
    center = np.array([0.5 * width, 0.5 * height])
    step = float(context.get("sigma", 3e-7) or 3e-7)
    changed_indices = []
    for idx in small:
        centers[idx] += rng.normal(0.0, 2.5 * step, size=2)
        changed_indices.append(int(idx))
    for idx in medium:
        direction = centers[idx] - center
        norm = max(float(np.linalg.norm(direction)), 1e-12)
        centers[idx] += direction / norm * rng.normal(0.0, step)
        changed_indices.append(int(idx))
    centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)
    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)
    metadata["blocks_used"].append("geometry:radius_group_redistribution")
    metadata["strategy_family"] = "radius_group_redistribution"
    metadata["operator_name"] = "radius_group_redistribution"
    metadata["intended_contact_change"] = "small circles explore, large circles remain stable"
    metadata["intended_boundary_change"] = "possible"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"q1": float(q1), "q2": float(q2), "step": step}
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
    max_steps = int(context.get("max_refine_steps", 4) or 4)
    changed_indices = []
    for _ in range(max_steps):
        centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)
        centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)
    metadata["blocks_used"].append("refine:contact_graph_preserving_refine")
    metadata["strategy_family"] = "contact_graph_preserving_refine"
    metadata["operator_name"] = "contact_graph_preserving_refine"
    metadata["intended_contact_change"] = "preserve active graph while LP reallocates radii"
    metadata["intended_boundary_change"] = "preserve"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"max_steps": max_steps}
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
