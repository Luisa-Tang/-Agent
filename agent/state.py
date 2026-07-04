"""Explicit per-iteration Agent state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
