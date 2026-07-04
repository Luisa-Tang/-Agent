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
    box = np.array([width, height], dtype=float)
    margins = np.minimum.reduce([centers[:, 0], box[0] - centers[:, 0], centers[:, 1], box[1] - centers[:, 1]])
    idx = np.argsort(margins)[: max(2, min(6, len(margins)))]
    changed_indices = []
    if len(idx) >= 2:
        a = int(idx[int(rng.integers(0, len(idx)))])
        b = int(idx[int(rng.integers(0, len(idx)))])
        if a != b:
            if rng.random() < 0.5:
                centers[a, 0], centers[b, 0] = centers[b, 0], centers[a, 0]
            else:
                centers[a, 1], centers[b, 1] = centers[b, 1], centers[a, 1]
            changed_indices = [a, b]
    centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)
    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)
    metadata["blocks_used"].append("geometry:boundary_pattern_swap")
    metadata["strategy_family"] = "boundary_pattern_swap"
    metadata["operator_name"] = "boundary_pattern_swap"
    metadata["intended_contact_change"] = "change boundary-neighbor assignments"
    metadata["intended_boundary_change"] = "swap active boundary positions"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"candidate_count": int(len(idx))}
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
