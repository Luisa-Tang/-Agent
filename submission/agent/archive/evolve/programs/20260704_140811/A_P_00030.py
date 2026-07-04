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
    delta = float(context.get("aspect_delta", context.get("target_delta", 1e-7)) or 1e-7)
    direction = float(context.get("width_direction", 1.0) or 1.0)
    width = float(np.clip(width + direction * delta, 0.28, 1.72))
    height = 2.0 - width
    x_scale = width / max(old_width, 1e-12)
    y_scale = height / max(old_height, 1e-12)
    centers[:, 0] *= x_scale
    centers[:, 1] *= y_scale
    metadata["blocks_used"].append("aspect:aspect_ratio_sweep_local")
    metadata["operator_name"] = "aspect_ratio_sweep_local"
    metadata["parameters"] = {"aspect_delta": delta, "width_direction": direction}
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
