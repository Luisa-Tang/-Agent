"""Geometry and local optimization helpers for circle-packing candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


TASK_SPECS = {
    "A": {"n": 21, "target": 2.365840},
    "B": {"n": 26, "target": 2.635990},
}


@dataclass
class PackingData:
    task: str
    centers: np.ndarray
    radii: np.ndarray
    width: Optional[float] = None
    height: Optional[float] = None

    @property
    def sum_radii(self) -> float:
        return float(np.sum(self.radii))

    @property
    def score(self) -> float:
        return self.sum_radii / float(TASK_SPECS[self.task]["target"])


def pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, 1)


def boundary_limits(task: str, centers: np.ndarray, width: Optional[float] = None,
                    height: Optional[float] = None) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if task == "A":
        if width is None or height is None:
            raise ValueError("Task A requires width and height")
        return np.minimum.reduce([
            centers[:, 0],
            width - centers[:, 0],
            centers[:, 1],
            height - centers[:, 1],
        ])
    return np.minimum.reduce([
        centers[:, 0],
        1.0 - centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 1],
    ])


def clip_centers(task: str, centers: np.ndarray, width: Optional[float] = None,
                 height: Optional[float] = None, eps: float = 1e-10) -> np.ndarray:
    centers = np.asarray(centers, dtype=float).copy()
    if task == "A":
        centers[:, 0] = np.clip(centers[:, 0], eps, float(width) - eps)
        centers[:, 1] = np.clip(centers[:, 1], eps, float(height) - eps)
    else:
        centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
        centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return centers


def equal_safe_radius(task: str, centers: np.ndarray, width: Optional[float] = None,
                      height: Optional[float] = None, margin: float = 1e-6) -> float:
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    lim = float(np.min(boundary_limits(task, centers, width, height)))
    if n > 1:
        i, j = pair_indices(n)
        d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2, axis=1))
        lim = min(lim, 0.5 * float(np.min(d)))
    return max(0.0, lim - margin)


def solve_radii_lp(task: str, centers: np.ndarray, width: Optional[float] = None,
                   height: Optional[float] = None, margin: float = 1e-7) -> np.ndarray:
    """Maximize sum radii for fixed centers using a linear program.

    Falls back to a conservative equal-radius construction if scipy is missing
    or the LP solver cannot prove success.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    bnd = boundary_limits(task, centers, width, height) - margin
    bnd = np.maximum(bnd, 0.0)
    fallback = np.full(n, equal_safe_radius(task, centers, width, height, margin))
    try:
        from scipy.optimize import linprog
    except Exception:
        return fallback

    rows = []
    rhs = []
    for k in range(n):
        row = np.zeros(n)
        row[k] = 1.0
        rows.append(row)
        rhs.append(float(bnd[k]))

    i_idx, j_idx = pair_indices(n)
    d = np.sqrt(np.sum((centers[i_idx] - centers[j_idx]) ** 2, axis=1))
    for i, j, dist in zip(i_idx, j_idx, d):
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        rows.append(row)
        rhs.append(max(0.0, float(dist) - margin))

    res = linprog(
        c=-np.ones(n),
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs),
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if not res.success or res.x is None or not np.isfinite(res.x).all():
        return fallback
    return np.maximum(np.asarray(res.x, dtype=float), 0.0)


def validate_packing(task: str, centers: np.ndarray, radii: np.ndarray,
                     width: Optional[float] = None, height: Optional[float] = None,
                     tol: float = 1e-9) -> Tuple[bool, str]:
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = int(TASK_SPECS[task]["n"])
    if centers.shape != (n, 2):
        return False, f"centers.shape {centers.shape} != ({n}, 2)"
    if radii.shape != (n,):
        return False, f"radii.shape {radii.shape} != ({n},)"
    if not (np.isfinite(centers).all() and np.isfinite(radii).all()):
        return False, "nonfinite"
    if not (radii >= 0.0).all():
        return False, "negative"
    if task == "A":
        if width is None or height is None:
            return False, "missing width/height"
        if abs((float(width) + float(height)) - 2.0) > 1e-6:
            return False, "perimeter"
        limits = boundary_limits(task, centers, width, height)
    else:
        limits = boundary_limits(task, centers)
    if np.any(radii - limits > tol):
        return False, "outside"
    i, j = pair_indices(n)
    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2, axis=1))
    if np.any(radii[i] + radii[j] - dist > tol):
        return False, "overlap"
    return True, "none"


def safety_repair(task: str, centers: np.ndarray, radii: np.ndarray,
                  width: Optional[float] = None, height: Optional[float] = None,
                  safety: float = 2e-9) -> Tuple[np.ndarray, np.ndarray]:
    centers = clip_centers(task, centers, width, height, eps=safety)
    radii = np.asarray(radii, dtype=float).copy()
    radii = np.maximum(radii, 0.0)
    radii = np.minimum(radii, np.maximum(0.0, boundary_limits(task, centers, width, height) - safety))
    n = len(radii)
    for _ in range(4):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                limit = max(0.0, d - safety)
                total = radii[i] + radii[j]
                if total > limit and total > 0.0:
                    factor = limit / total
                    radii[i] *= factor
                    radii[j] *= factor
                    changed = True
        if not changed:
            break
    return centers, np.maximum(radii, 0.0)


def grid_centers(n: int, width: float, height: float, rows: int, cols: int,
                 stagger: bool = False, jitter: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    pts = []
    for row in range(rows):
        for col in range(cols):
            if len(pts) >= n:
                break
            x = (col + 0.5) * width / cols
            y = (row + 0.5) * height / rows
            if stagger and row % 2 == 1:
                x += 0.5 * width / cols
                if x > width:
                    x -= width / cols
            pts.append([x, y])
        if len(pts) >= n:
            break
    arr = np.asarray(pts, dtype=float)
    if jitter and rng is not None:
        arr[:, 0] += rng.uniform(-jitter, jitter, size=len(arr)) * width / cols
        arr[:, 1] += rng.uniform(-jitter, jitter, size=len(arr)) * height / rows
        arr = clip_centers("A", arr, width, height, eps=1e-6)
    return arr


def row_count_centers(n: int, width: float, height: float, rows: int,
                      stagger: bool = True, rng: Optional[np.random.Generator] = None,
                      jitter: float = 0.0) -> np.ndarray:
    base = n // rows
    rem = n % rows
    counts = [base + (1 if i < rem else 0) for i in range(rows)]
    if stagger:
        counts = sorted(counts, reverse=True)
    pts = []
    for row, count in enumerate(counts):
        y = (row + 0.5) * height / rows
        xs = np.linspace(width / (2 * count), width - width / (2 * count), count)
        if stagger and row % 2 == 1 and count > 1:
            xs = xs + width / (4 * count)
            xs = np.clip(xs, width / (2 * count), width - width / (2 * count))
        for x in xs:
            pts.append([float(x), float(y)])
    arr = np.asarray(pts[:n], dtype=float)
    if jitter and rng is not None:
        arr[:, 0] += rng.uniform(-jitter, jitter, size=len(arr)) * width / max(counts)
        arr[:, 1] += rng.uniform(-jitter, jitter, size=len(arr)) * height / rows
        arr = clip_centers("A", arr, width, height, eps=1e-6)
    return arr


def structured_initial_data(task: str, variant: str, seed: int = 42,
                            width_hint: Optional[float] = None) -> PackingData:
    rng = np.random.default_rng(seed)
    n = int(TASK_SPECS[task]["n"])
    if task == "A":
        width = float(width_hint if width_hint is not None else 1.0)
        width = float(np.clip(width, 0.35, 1.65))
        height = 2.0 - width
        if variant == "baseline_safe_grid":
            centers = grid_centers(n, width, height, rows=5, cols=5)
        elif variant == "hexagonal_or_staggered_initialization":
            rows = 5 if height >= 0.75 else 4
            centers = row_count_centers(n, width, height, rows=rows, stagger=True)
        elif variant == "random_jittered_grid":
            centers = grid_centers(n, width, height, rows=5, cols=5, stagger=True,
                                   jitter=0.28, rng=rng)
        else:
            rows = int(round(math.sqrt(n * height / max(width, 1e-12))))
            rows = int(np.clip(rows, 3, 7))
            cols = int(math.ceil(n / rows))
            centers = grid_centers(n, width, height, rows=rows, cols=cols, stagger=True,
                                   jitter=0.15, rng=rng)
        centers = clip_centers(task, centers, width, height)
        radii = solve_radii_lp(task, centers, width, height, margin=1e-7)
        centers, radii = safety_repair(task, centers, radii, width, height)
        return PackingData(task=task, centers=centers, radii=radii, width=width, height=height)

    width = height = 1.0
    if variant == "baseline_safe_grid":
        centers = grid_centers(n, width, height, rows=5, cols=6)
    elif variant == "hexagonal_or_staggered_initialization":
        centers = row_count_centers(n, width, height, rows=5, stagger=True)
    elif variant == "random_jittered_grid":
        centers = grid_centers(n, width, height, rows=5, cols=6, stagger=True,
                               jitter=0.30, rng=rng)
    else:
        rows = int(np.clip(round(math.sqrt(n)), 4, 7))
        cols = int(math.ceil(n / rows))
        centers = grid_centers(n, width, height, rows=rows, cols=cols, stagger=True,
                               jitter=0.18, rng=rng)
    centers = clip_centers(task, centers)
    radii = solve_radii_lp(task, centers, margin=1e-7)
    centers, radii = safety_repair(task, centers, radii)
    return PackingData(task=task, centers=centers, radii=radii)


def summarize_margins(task: str, centers: np.ndarray, radii: np.ndarray,
                      width: Optional[float] = None, height: Optional[float] = None) -> dict:
    bmin = float(np.min(boundary_limits(task, centers, width, height) - radii))
    n = len(radii)
    if n <= 1:
        omin = float("inf")
    else:
        i, j = pair_indices(n)
        dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2, axis=1))
        omin = float(np.min(dist - radii[i] - radii[j]))
    return {"boundary_margin": bmin, "overlap_margin": omin}


def safety_metrics(data: PackingData) -> dict:
    """Return trace-friendly safety and score metrics for a candidate."""
    margins = summarize_margins(data.task, data.centers, data.radii, data.width, data.height)
    metrics = {
        "min_pairwise_margin": float(margins["overlap_margin"]),
        "min_boundary_margin": float(margins["boundary_margin"]),
        "sum_radii": data.sum_radii,
        "score": data.score,
        "width": float(data.width) if data.task == "A" and data.width is not None else None,
        "height": float(data.height) if data.task == "A" and data.height is not None else None,
    }
    return metrics
