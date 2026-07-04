"""Gap candidates for risky circle-packing structure search."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GapCandidate:
    gap_id: str
    point: List[float]
    max_insertable_radius: float
    nearest_circles: List[int]
    boundary_sides: List[str]
    gap_score: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_gap_candidates_task_b(centers: np.ndarray, radii: np.ndarray,
                                  rng: Optional[np.random.Generator] = None,
                                  random_samples: int = 48) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in _compute_gap_candidates("B", centers, radii, 1.0, 1.0, rng, random_samples)]


def compute_gap_candidates_task_a(centers: np.ndarray, radii: np.ndarray,
                                  width: float, height: float,
                                  rng: Optional[np.random.Generator] = None,
                                  random_samples: int = 48) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in _compute_gap_candidates("A", centers, radii, width, height, rng, random_samples)]


def score_gap_candidate(point: Sequence[float], centers: np.ndarray, radii: np.ndarray,
                        width: float, height: float, source: str = "manual") -> Dict[str, Any]:
    candidate = _score_point(np.asarray(point, dtype=float), centers, radii, width, height, source, "manual")
    return candidate.to_dict()


def top_k_gaps(candidates: Iterable[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    return sorted(candidates, key=lambda item: float(item.get("gap_score") or -1.0), reverse=True)[: max(0, int(k))]


def refill_small_circles(centers: np.ndarray, radii: np.ndarray,
                         width: float, height: float,
                         removed_indices: Sequence[int],
                         rng: Optional[np.random.Generator] = None,
                         mode: str = "destroy_repair",
                         random_samples: int = 64) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = rng or np.random.default_rng(0)
    centers = np.asarray(centers, dtype=float).copy()
    radii = np.asarray(radii, dtype=float)
    removed = [int(i) for i in removed_indices if 0 <= int(i) < len(radii)]
    if not removed:
        return centers, {
            "gap_ids": [],
            "removed_indices": [],
            "gap_sources": [],
            "small_circle_reassigned": False,
        }

    fixed_mask = np.ones(len(radii), dtype=bool)
    fixed_mask[removed] = False
    fixed_centers = centers[fixed_mask]
    fixed_radii = radii[fixed_mask]
    candidates = _compute_gap_candidates("B", fixed_centers, fixed_radii, width, height, rng, random_samples)
    gaps = top_k_gaps([item.to_dict() for item in candidates], k=max(6, 3 * len(removed)))
    if not gaps:
        for idx in removed:
            centers[idx] = rng.uniform([0.05 * width, 0.05 * height], [0.95 * width, 0.95 * height])
        return centers, {
            "gap_ids": [],
            "removed_indices": removed,
            "gap_sources": ["random_fallback"],
            "small_circle_reassigned": True,
        }

    used_gap_ids = []
    gap_sources = []
    largest_removed = sorted(removed, key=lambda i: float(radii[i]), reverse=True)
    for rank, idx in enumerate(largest_removed):
        gap = gaps[rank % len(gaps)]
        point = np.asarray(gap.get("point"), dtype=float)
        jitter_scale = max(1e-7, 0.03 * max(float(gap.get("max_insertable_radius") or 0.0), float(radii[idx])))
        centers[idx] = point + rng.normal(0.0, jitter_scale, size=2)
        used_gap_ids.append(str(gap.get("gap_id")))
        gap_sources.append(str(gap.get("source")))
    return centers, {
        "gap_ids": used_gap_ids,
        "removed_indices": removed,
        "gap_sources": gap_sources,
        "small_circle_reassigned": True,
        "top_gap_score": float(gaps[0].get("gap_score") or 0.0),
        "top_gap_radius": float(gaps[0].get("max_insertable_radius") or 0.0),
        "mode": mode,
    }


def _compute_gap_candidates(task: str, centers: np.ndarray, radii: np.ndarray,
                            width: float, height: float,
                            rng: Optional[np.random.Generator],
                            random_samples: int) -> List[GapCandidate]:
    rng = rng or np.random.default_rng(0)
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    points: List[Tuple[np.ndarray, str]] = []
    n = len(radii)

    for i in range(n):
        for j in range(i + 1, n):
            midpoint = 0.5 * (centers[i] + centers[j])
            direction = centers[j] - centers[i]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                normal = np.array([-direction[1], direction[0]]) / norm
                scale = 0.5 * max(float(radii[i] + radii[j]), 1e-6)
                points.append((midpoint + scale * normal, "two_circle_gap"))
                points.append((midpoint - scale * normal, "two_circle_gap"))

    if n >= 3:
        order = np.argsort(radii)[: min(n, 12)]
        for a_pos, i in enumerate(order):
            for j in order[a_pos + 1:]:
                for k in order[a_pos + 2:]:
                    triangle = np.vstack([centers[int(i)], centers[int(j)], centers[int(k)]])
                    points.append((np.mean(triangle, axis=0), "three_circle_centroid"))
                    break

    for i in range(n):
        x, y = centers[i]
        r = max(float(radii[i]), 1e-6)
        points.extend(
            [
                (np.array([x, y + 1.8 * r]), "circle_boundary_gap"),
                (np.array([x, y - 1.8 * r]), "circle_boundary_gap"),
                (np.array([x + 1.8 * r, y]), "circle_boundary_gap"),
                (np.array([x - 1.8 * r, y]), "circle_boundary_gap"),
            ]
        )

    xs = np.linspace(0.08 * width, 0.92 * width, 5)
    ys = np.linspace(0.08 * height, 0.92 * height, 5)
    for x in xs:
        points.append((np.array([x, 0.03 * height]), "boundary_long_gap"))
        points.append((np.array([x, 0.97 * height]), "boundary_long_gap"))
    for y in ys:
        points.append((np.array([0.03 * width, y]), "boundary_long_gap"))
        points.append((np.array([0.97 * width, y]), "boundary_long_gap"))

    for _ in range(max(0, int(random_samples))):
        points.append((rng.uniform([0.02 * width, 0.02 * height], [0.98 * width, 0.98 * height]), "random_empty_sample"))

    candidates: List[GapCandidate] = []
    seen = set()
    for idx, (point, source) in enumerate(points):
        point = np.asarray(point, dtype=float)
        point[0] = float(np.clip(point[0], 1e-10, width - 1e-10))
        point[1] = float(np.clip(point[1], 1e-10, height - 1e-10))
        key = (round(float(point[0]), 9), round(float(point[1]), 9), source)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_score_point(point, centers, radii, width, height, source, f"gap_{idx:04d}"))
    candidates.sort(key=lambda item: item.gap_score, reverse=True)
    return candidates


def _score_point(point: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                 width: float, height: float, source: str, gap_id: str) -> GapCandidate:
    if len(radii):
        distances = np.sqrt(np.sum((centers - point) ** 2, axis=1)) - radii
        nearest_order = np.argsort(distances)[: min(4, len(distances))]
        circle_clearance = float(np.min(distances))
        nearest = [int(i) for i in nearest_order]
    else:
        circle_clearance = float("inf")
        nearest = []
    boundary_values = {
        "left": float(point[0]),
        "right": float(width - point[0]),
        "bottom": float(point[1]),
        "top": float(height - point[1]),
    }
    boundary_clearance = min(boundary_values.values())
    max_radius = max(0.0, min(circle_clearance, boundary_clearance))
    side_threshold = max(1e-9, 1.5 * max_radius)
    sides = [name for name, value in boundary_values.items() if value <= side_threshold]
    centrality = min(point[0] / max(width, 1e-12), 1.0 - point[0] / max(width, 1e-12),
                    point[1] / max(height, 1e-12), 1.0 - point[1] / max(height, 1e-12))
    source_bonus = {
        "three_circle_centroid": 0.08,
        "two_circle_gap": 0.06,
        "boundary_long_gap": 0.05,
        "circle_boundary_gap": 0.04,
        "random_empty_sample": 0.0,
    }.get(source, 0.0)
    score = max_radius + 0.02 * max(0.0, centrality) + source_bonus
    return GapCandidate(
        gap_id=str(gap_id),
        point=[float(point[0]), float(point[1])],
        max_insertable_radius=float(max_radius),
        nearest_circles=nearest,
        boundary_sides=sides,
        gap_score=float(score),
        source=source,
    )
