"""Contact-graph neighborhood refinement for breakthrough search."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from candidate_generators import GeneratedCandidate, solution_code_for
from contact_graph import active_boundary_pattern, active_contact_graph, summarize_contact_graph
from geometry_utils import PackingData, TASK_SPECS, clip_centers, safety_repair, solve_radii_lp, validate_packing


DELTAS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6]


def contact_graph_feasibility_refine(task: str, parent_data: PackingData,
                                     iteration: int = 0,
                                     seed: int = 0,
                                     max_candidates: int = 8,
                                     deltas: Optional[List[float]] = None) -> List[GeneratedCandidate]:
    task = task.upper()
    deltas = list(deltas or DELTAS)
    candidates: List[GeneratedCandidate] = []
    rng = np.random.default_rng(int(seed) + 10007 * (iteration + 1))
    contact_summary = summarize_contact_graph(
        task, parent_data.centers, parent_data.radii, parent_data.width, parent_data.height, tolerance=5e-8
    )
    for index, delta in enumerate(deltas[:max_candidates]):
        if index == 0 and iteration < 2:
            data, diag = _least_squares_target(task, parent_data, float(delta), rng)
        else:
            data = _lp_fallback(task, parent_data, jitter=min(3e-7, max(1e-9, float(delta) * 0.15)), rng=rng)
            diag = {"optimizer": "fixed_center_lp_jitter", "success": True, "message": "cheap batch neighbor"}
        diag.update(
            {
                "source": "contact_graph_feasibility_refine",
                "strategy_family": "contact_graph_refinement",
                "parent_sum_radii": parent_data.sum_radii,
                "target_sum_radii": parent_data.sum_radii + float(delta),
                "delta": float(delta),
                "parent_contact_graph": contact_summary,
            }
        )
        valid, message = validate_packing(task, data.centers, data.radii, data.width, data.height, tol=1e-8)
        diag["internal_valid"] = valid
        diag["internal_message"] = message
        diag["sum_radii"] = data.sum_radii
        diag["score_estimate"] = data.score
        diag["contact_graph"] = summarize_contact_graph(task, data.centers, data.radii, data.width, data.height, tolerance=5e-8)
        code = solution_code_for(data, strategy="contact_graph_feasibility_refine", diagnostics=diag)
        candidates.append(
            GeneratedCandidate(
                task=task,
                strategy="contact_graph_feasibility_refine",
                code=code,
                data=data,
                diagnostics=diag,
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def _least_squares_target(task: str, parent: PackingData, delta: float,
                          rng: np.random.Generator) -> tuple[PackingData, Dict[str, object]]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        data = _lp_fallback(task, parent, jitter=0.0)
        return data, {"optimizer": "lp_fallback", "exception": repr(exc)}

    n = int(TASK_SPECS[task]["n"])
    centers0 = np.asarray(parent.centers, dtype=float)
    radii0 = np.asarray(parent.radii, dtype=float)
    active_edges = active_contact_graph(centers0, radii0, tolerance=5e-7)
    target_sum = float(parent.sum_radii + delta)
    jitter = min(2e-7, max(2e-9, abs(delta) * 0.2))
    centers_start = centers0 + rng.normal(0.0, jitter, size=centers0.shape)
    radii_start = np.maximum(0.0, radii0 + delta / max(n, 1))

    if task == "A":
        width0 = float(parent.width)
        height0 = 2.0 - width0
        z0 = np.concatenate([[width0], centers_start.reshape(-1), radii_start])
    else:
        z0 = np.concatenate([centers_start.reshape(-1), radii_start])

    def unpack(z):
        if task == "A":
            width = float(np.clip(z[0], 0.25, 1.75))
            height = 2.0 - width
            centers = np.asarray(z[1:1 + 2 * n], dtype=float).reshape(n, 2)
            radii = np.asarray(z[1 + 2 * n:1 + 3 * n], dtype=float)
            return width, height, centers, radii
        centers = np.asarray(z[:2 * n], dtype=float).reshape(n, 2)
        radii = np.asarray(z[2 * n:3 * n], dtype=float)
        return 1.0, 1.0, centers, radii

    def residuals(z):
        width, height, centers, radii = unpack(z)
        centers = clip_centers(task, centers, width, height, eps=1e-10)
        radii = np.maximum(radii, 0.0)
        residual = []
        residual.append((float(np.sum(radii)) - target_sum) * 30.0)
        if task == "A":
            residual.append((width - float(parent.width)) * 0.05)
        residual.extend(((centers - centers0).reshape(-1) * 0.02).tolist())
        residual.extend(((radii - radii_start) * 0.01).tolist())
        for i, j in active_edges:
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            residual.append((dist - radii[i] - radii[j]) * 10.0)
        boundary = _boundary_values(task, centers, radii, width, height)
        pair_violations = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(centers[i] - centers[j]))
                pair_violations.append(min(0.0, dist - radii[i] - radii[j] - 1e-10) * 80.0)
        residual.extend(pair_violations)
        residual.extend((np.minimum(boundary - 1e-10, 0.0).reshape(-1) * 80.0).tolist())
        residual.extend((np.minimum(radii, 0.0) * 80.0).tolist())
        return np.asarray(residual, dtype=float)

    lower = np.full_like(z0, -np.inf, dtype=float)
    upper = np.full_like(z0, np.inf, dtype=float)
    if task == "A":
        lower[0] = 0.25
        upper[0] = 1.75
        lower[1:1 + 2 * n] = 0.0
        upper[1:1 + 2 * n] = 1.8
        lower[1 + 2 * n:] = 0.0
        upper[1 + 2 * n:] = 0.8
    else:
        lower[:2 * n] = 0.0
        upper[:2 * n] = 1.0
        lower[2 * n:] = 0.0
        upper[2 * n:] = 0.6

    success = False
    cost = None
    message = ""
    result_x = z0
    try:
        res = least_squares(
            residuals,
            z0,
            bounds=(lower, upper),
            max_nfev=45,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        if res.x is not None and np.isfinite(res.x).all():
            result_x = np.asarray(res.x, dtype=float)
        success = bool(res.success)
        cost = float(res.cost)
        message = str(res.message)
    except Exception as exc:
        message = repr(exc)

    width, height, centers, _radii = unpack(result_x)
    centers = clip_centers(task, centers, width, height, eps=1e-10)
    radii = solve_radii_lp(task, centers, width if task == "A" else None, height if task == "A" else None, margin=0.0)
    centers, radii = safety_repair(task, centers, radii, width if task == "A" else None, height if task == "A" else None, safety=2e-10)
    data = PackingData(task=task, centers=centers, radii=radii, width=width if task == "A" else None, height=height if task == "A" else None)
    return data, {
        "optimizer": "scipy.optimize.least_squares",
        "success": success,
        "message": message,
        "cost": cost,
        "active_edge_count": len(active_edges),
        "active_boundary_pattern": active_boundary_pattern(task, centers0, radii0, parent.width, parent.height, tolerance=5e-7),
    }


def _lp_fallback(task: str, parent: PackingData, jitter: float,
                 rng: Optional[np.random.Generator] = None) -> PackingData:
    centers = np.asarray(parent.centers, dtype=float).copy()
    if jitter > 0.0 and rng is not None:
        centers += rng.normal(0.0, jitter, size=centers.shape)
    width = parent.width
    height = parent.height
    if task == "A" and width is not None:
        width = float(np.clip(float(width), 0.25, 1.75))
        height = 2.0 - width
    centers = clip_centers(task, centers, width, height, eps=1e-10)
    radii = solve_radii_lp(task, centers, width, height, margin=0.0)
    centers, radii = safety_repair(task, centers, radii, width, height, safety=2e-10)
    return PackingData(task=task, centers=centers, radii=radii, width=width, height=height)


def _boundary_values(task: str, centers: np.ndarray, radii: np.ndarray,
                     width: float, height: float) -> np.ndarray:
    if task == "A":
        w = float(width)
        h = float(height)
    else:
        w = 1.0
        h = 1.0
    return np.vstack(
        [
            centers[:, 0] - radii,
            w - centers[:, 0] - radii,
            centers[:, 1] - radii,
            h - centers[:, 1] - radii,
        ]
    ).T
