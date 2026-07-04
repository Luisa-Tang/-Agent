"""Lightweight novelty rejection for self-evolved programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np

from geometry_utils import PackingData


@dataclass
class NoveltyDecision:
    accepted: bool
    novelty_score: float
    failure_type: str = "none"
    reasons: Dict[str, Any] = field(default_factory=dict)


class NoveltyFilter:
    def __init__(self, threshold: float = 0.25):
        self.threshold = float(threshold)
        self.code_hashes: Set[str] = set()
        self.contact_hashes: Set[str] = set()
        self.boundary_patterns: Set[str] = set()
        self.recent_strategy_families = []
        self.rejected_count = 0

    def remember(self, code_hash: Optional[str], contact_hash: Optional[str],
                 boundary_pattern: Optional[str], strategy_family: Optional[str],
                 block_hashes: Optional[Dict[str, str]] = None) -> None:
        if code_hash:
            self.code_hashes.add(str(code_hash))
        for value in (block_hashes or {}).values():
            if value:
                self.code_hashes.add(str(value))
        if contact_hash:
            self.contact_hashes.add(str(contact_hash))
        if boundary_pattern:
            self.boundary_patterns.add(str(boundary_pattern))
        if strategy_family:
            self.recent_strategy_families.append(str(strategy_family))
            self.recent_strategy_families = self.recent_strategy_families[-8:]

    def judge(self, code_hash: str, contact_hash: str, boundary_pattern: str,
              strategy_family: str, data: PackingData,
              parent_data: Optional[PackingData],
              parent_contact_hash: Optional[str] = None,
              parent_boundary_pattern: Optional[str] = None,
              code_block_hash: Optional[str] = None,
              island_name: str = "safe_polish",
              small_circle_reassigned: bool = False,
              aspect_ratio_bucket_changed: bool = False,
              centers_rmsd_threshold: float = 3e-6,
              radii_l2_threshold: float = 1e-7) -> NoveltyDecision:
        code_new = (code_block_hash or code_hash) not in self.code_hashes
        contact_graph_changed = bool(parent_contact_hash) and str(parent_contact_hash) != str(contact_hash)
        boundary_pattern_changed = bool(parent_boundary_pattern) and str(parent_boundary_pattern) != str(boundary_pattern)
        contact_new = contact_hash not in self.contact_hashes
        boundary_new = boundary_pattern not in self.boundary_patterns
        rmsd = centers_rmsd(data, parent_data)
        radii_l2 = sorted_radii_l2(data, parent_data)
        centers_rmsd_novelty = min(1.0, max(0.0, rmsd / 3e-6))
        radius_distribution_novelty = min(1.0, max(0.0, radii_l2 / 3e-7))
        repeated_family = len(self.recent_strategy_families) >= 3 and all(
            item == strategy_family for item in self.recent_strategy_families[-3:]
        )
        contact_graph_novelty = 1.0 if (contact_new or contact_graph_changed) else 0.0
        boundary_pattern_novelty = 1.0 if (boundary_new or boundary_pattern_changed) else 0.0
        strategy_family_novelty = 0.0 if repeated_family else 1.0
        score = (
            0.25 * (1.0 if code_new else 0.0)
            + 0.25 * contact_graph_novelty
            + 0.20 * boundary_pattern_novelty
            + 0.15 * centers_rmsd_novelty
            + 0.10 * radius_distribution_novelty
            + 0.05 * strategy_family_novelty
        )
        hard_equivalent = (
            not contact_graph_changed
            and not boundary_pattern_changed
            and rmsd < float(centers_rmsd_threshold)
            and radii_l2 < float(radii_l2_threshold)
        )
        risky_mode = str(island_name or "safe_polish") in {"risky_structure", "aspect_ratio_island"}
        risky_has_structure_signal = (
            contact_graph_changed
            or boundary_pattern_changed
            or rmsd > float(centers_rmsd_threshold)
            or bool(small_circle_reassigned)
            or bool(aspect_ratio_bucket_changed)
        )
        accepted = score >= self.threshold
        rejection_reason = "none"
        if hard_equivalent:
            accepted = False
            rejection_reason = "geometry_hard_equivalent_to_parent"
        elif risky_mode and not risky_has_structure_signal:
            accepted = False
            rejection_reason = "risky_structure_without_structure_signal"
        elif score < self.threshold:
            if not contact_graph_novelty and not boundary_pattern_novelty and centers_rmsd_novelty < 0.05:
                rejection_reason = "geometry_equivalent_to_archive_or_parent"
            elif not code_new:
                rejection_reason = "code_block_repeated"
            else:
                rejection_reason = "novelty_below_threshold"
        reasons = {
            "code_block_novelty": 1.0 if code_new else 0.0,
            "contact_graph_novelty": contact_graph_novelty,
            "boundary_pattern_novelty": boundary_pattern_novelty,
            "centers_rmsd_novelty": centers_rmsd_novelty,
            "radius_distribution_novelty": radius_distribution_novelty,
            "strategy_family_novelty": strategy_family_novelty,
            "code_hash_new": code_new,
            "contact_graph_hash_new": contact_new,
            "boundary_pattern_new": boundary_new,
            "contact_graph_changed": contact_graph_changed,
            "boundary_pattern_changed": boundary_pattern_changed,
            "centers_rmsd": rmsd,
            "centers_rmsd_to_parent": rmsd,
            "sorted_radii_l2_to_parent": radii_l2,
            "strategy_family_repeated": repeated_family,
            "island_name": str(island_name or "safe_polish"),
            "small_circle_reassigned": bool(small_circle_reassigned),
            "aspect_ratio_bucket_changed": bool(aspect_ratio_bucket_changed),
            "hard_equivalent_gate": hard_equivalent,
            "risky_structure_signal": risky_has_structure_signal,
            "novelty_rejection_reason": rejection_reason,
            "threshold": self.threshold,
        }
        if not accepted:
            self.rejected_count += 1
        return NoveltyDecision(
            accepted=accepted,
            novelty_score=float(score),
            failure_type="none" if accepted else "rejected_novelty",
            reasons=reasons,
        )


def centers_rmsd(data: PackingData, parent_data: Optional[PackingData]) -> float:
    if parent_data is None:
        return 1.0
    a = np.asarray(data.centers, dtype=float)
    b = np.asarray(parent_data.centers, dtype=float)
    if a.shape != b.shape:
        return 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sorted_radii_l2(data: PackingData, parent_data: Optional[PackingData]) -> float:
    if parent_data is None:
        return 1.0
    a = np.sort(np.asarray(data.radii, dtype=float))
    b = np.sort(np.asarray(parent_data.radii, dtype=float))
    if a.shape != b.shape:
        return 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)))
