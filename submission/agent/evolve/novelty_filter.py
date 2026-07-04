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
                 boundary_pattern: Optional[str], strategy_family: Optional[str]) -> None:
        if code_hash:
            self.code_hashes.add(str(code_hash))
        if contact_hash:
            self.contact_hashes.add(str(contact_hash))
        if boundary_pattern:
            self.boundary_patterns.add(str(boundary_pattern))
        if strategy_family:
            self.recent_strategy_families.append(str(strategy_family))
            self.recent_strategy_families = self.recent_strategy_families[-8:]

    def judge(self, code_hash: str, contact_hash: str, boundary_pattern: str,
              strategy_family: str, data: PackingData,
              parent_data: Optional[PackingData]) -> NoveltyDecision:
        code_new = code_hash not in self.code_hashes
        contact_new = contact_hash not in self.contact_hashes
        boundary_new = boundary_pattern not in self.boundary_patterns
        rmsd = centers_rmsd(data, parent_data)
        rmsd_component = min(0.25, max(0.0, rmsd / 3e-6) * 0.25)
        repeated_family = len(self.recent_strategy_families) >= 3 and all(
            item == strategy_family for item in self.recent_strategy_families[-3:]
        )
        score = 0.0
        score += 0.25 if code_new else 0.0
        score += 0.25 if contact_new else 0.0
        score += 0.15 if boundary_new else 0.0
        score += rmsd_component
        score += 0.10 if not repeated_family else 0.0
        reasons = {
            "code_hash_new": code_new,
            "contact_graph_hash_new": contact_new,
            "boundary_pattern_new": boundary_new,
            "centers_rmsd": rmsd,
            "rmsd_component": rmsd_component,
            "strategy_family_repeated": repeated_family,
            "threshold": self.threshold,
        }
        accepted = score >= self.threshold
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
