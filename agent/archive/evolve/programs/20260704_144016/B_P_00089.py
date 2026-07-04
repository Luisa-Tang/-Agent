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
    n = len(radii)
    changed_indices = []
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                local_degree = 1.0
                stress = float(radii[i] + radii[j]) / max(d, 1e-12) / local_degree
                pairs.append((abs(d - float(radii[i] + radii[j])), stress, i, j))
        pairs.sort(key=lambda item: (item[0], item[1]))
        _m, _stress, i, j = pairs[int(context.get("pair_rank", 0) or 0) % len(pairs)]
        direction = centers[i] - centers[j]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction = direction / norm
            eps = float(context.get("edge_break_eps", context.get("sigma", 1e-5)) or 1e-5)
            centers[i] += eps * direction
            centers[j] -= eps * direction
            changed_indices = [int(i), int(j)]
    # risky island intentionally allows pre-repair out-of-bounds before cascade repair
    metadata["blocks_used"].append("geometry:contact_edge_break_then_repair")
    metadata["strategy_family"] = "contact_edge_break_then_repair"
    metadata["operator_name"] = "contact_edge_break_then_repair"
    metadata["intended_contact_change"] = "break one low-value tight edge before LP repair"
    metadata["intended_boundary_change"] = "possible"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"edge_break_eps": float(context.get("edge_break_eps", context.get("sigma", 1e-5)) or 1e-5)}
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
