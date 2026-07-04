"""Explicit per-iteration Agent state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class GeoOptState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    seed: int
    archive_summary: Dict[str, Any]
    strategy_stats: Dict[str, Any]
    selected_strategy: Optional[str]
    decision_reason: str
    candidate_id: Optional[str]
    parent_candidate_id: Optional[str]
    eval_result: Optional[Dict[str, Any]]
    failure_type: Optional[str]
    best_candidate_id: Optional[str]
    best_score: float
    best_sum_radii: float
    next_action: str
    skills_used: List[str]
    artifacts: Dict[str, Any]


@dataclass
class AgentState:
    task: str
    iteration: int
    archive_summary: Dict[str, Any] = field(default_factory=dict)
    last_eval: Optional[Dict[str, Any]] = None
    selected_strategy: Optional[str] = None
    next_action: str = "observe"
    best_candidate_id: Optional[str] = None
    budget: Dict[str, Any] = field(default_factory=dict)
    skills_used: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""

    def snapshot(self, phase: str) -> Dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = phase
        return payload
