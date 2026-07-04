"""Contact graph diagnostics for circle-packing candidates."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


BOUNDARY_NAMES = ("left", "right", "bottom", "top")


def compute_pairwise_margins(centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(radii)
    margins = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            margin = float(np.linalg.norm(centers[i] - centers[j]) - radii[i] - radii[j])
            margins[i, j] = margin
            margins[j, i] = margin
    return margins


def compute_boundary_margins(task: str, centers: np.ndarray, radii: np.ndarray,
                             width: Optional[float] = None,
                             height: Optional[float] = None) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if task.upper() == "A":
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


def active_contact_graph(centers: np.ndarray, radii: np.ndarray,
                         tolerance: float = 1e-7) -> List[Tuple[int, int]]:
    margins = compute_pairwise_margins(centers, radii)
    n = len(radii)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(margins[i, j])) <= tolerance:
                edges.append((i, j))
    return edges


def active_edges_set(centers: np.ndarray, radii: np.ndarray,
                     tolerance: float = 1e-7) -> set[Tuple[int, int]]:
    return set(active_contact_graph(centers, radii, tolerance=tolerance))


def boundary_contacts_set(task: str, centers: np.ndarray, radii: np.ndarray,
                          width: Optional[float] = None,
                          height: Optional[float] = None,
                          tolerance: float = 1e-7) -> set[Tuple[int, str]]:
    margins = compute_boundary_margins(task, centers, radii, width, height)
    contacts = set()
    for circle_idx in range(margins.shape[0]):
        for side_idx, side in enumerate(BOUNDARY_NAMES):
            if abs(float(margins[circle_idx, side_idx])) <= tolerance:
                contacts.add((int(circle_idx), side))
    return contacts


def contact_graph_edit_distance(parent_summary: Dict[str, Any],
                                child_summary: Dict[str, Any]) -> Dict[str, Any]:
    parent_edges = {tuple(map(int, edge)) for edge in parent_summary.get("active_edges") or []}
    child_edges = {tuple(map(int, edge)) for edge in child_summary.get("active_edges") or []}
    added_edges = sorted(child_edges - parent_edges)
    removed_edges = sorted(parent_edges - child_edges)
    parent_boundary = _boundary_pattern_to_set(str(parent_summary.get("active_boundary_pattern") or "none"))
    child_boundary = _boundary_pattern_to_set(str(child_summary.get("active_boundary_pattern") or "none"))
    boundary_added = sorted(child_boundary - parent_boundary)
    boundary_removed = sorted(parent_boundary - child_boundary)
    return {
        "added_edges": [(int(i), int(j)) for i, j in added_edges],
        "removed_edges": [(int(i), int(j)) for i, j in removed_edges],
        "edge_edit_distance": int(len(added_edges) + len(removed_edges)),
        "boundary_added": [(int(i), str(side)) for i, side in boundary_added],
        "boundary_removed": [(int(i), str(side)) for i, side in boundary_removed],
        "boundary_pattern_edit_distance": int(len(boundary_added) + len(boundary_removed)),
    }


def contact_graph_hash(edges: List[Tuple[int, int]], prefix: int = 16) -> str:
    payload = json.dumps(sorted((int(i), int(j)) for i, j in edges), sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:prefix] if prefix else digest


def active_boundary_pattern(task: str, centers: np.ndarray, radii: np.ndarray,
                            width: Optional[float] = None,
                            height: Optional[float] = None,
                            tolerance: float = 1e-7) -> str:
    margins = compute_boundary_margins(task, centers, radii, width, height)
    parts = []
    for idx, name in enumerate(BOUNDARY_NAMES):
        active = [str(i) for i, value in enumerate(margins[:, idx]) if abs(float(value)) <= tolerance]
        if active:
            parts.append(f"{name}:{','.join(active)}")
    return "|".join(parts) if parts else "none"


def summarize_contact_graph(task: str, centers: np.ndarray, radii: np.ndarray,
                            width: Optional[float] = None,
                            height: Optional[float] = None,
                            tolerance: float = 1e-7) -> Dict[str, Any]:
    pairwise = compute_pairwise_margins(centers, radii)
    boundary = compute_boundary_margins(task, centers, radii, width, height)
    edges = active_contact_graph(centers, radii, tolerance=tolerance)
    boundary_pattern = active_boundary_pattern(task, centers, radii, width, height, tolerance)
    pair_values = pairwise[np.triu_indices(len(radii), 1)] if len(radii) > 1 else np.asarray([0.0])
    return {
        "active_edges": [(int(i), int(j)) for i, j in edges],
        "active_edge_count": len(edges),
        "contact_graph_hash": contact_graph_hash(edges),
        "active_boundary_pattern": boundary_pattern,
        "active_boundary_count": int(np.sum(np.abs(boundary) <= tolerance)),
        "min_pairwise_margin": float(np.min(pair_values)) if len(pair_values) else 0.0,
        "min_boundary_margin": float(np.min(boundary)) if boundary.size else 0.0,
        "median_pairwise_margin": float(np.median(pair_values)) if len(pair_values) else 0.0,
        "tolerance": float(tolerance),
    }


def _boundary_pattern_to_set(pattern: str) -> set[Tuple[int, str]]:
    if not pattern or pattern == "none":
        return set()
    contacts = set()
    for part in pattern.split("|"):
        if ":" not in part:
            continue
        side, values = part.split(":", 1)
        for value in values.split(","):
            value = value.strip()
            if not value:
                continue
            try:
                contacts.add((int(value), side))
            except ValueError:
                continue
    return contacts
