"""UCB-style operator bandit for GeoEvolve-lite."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


OPERATORS = [
    "parameter_mutation",
    "solver_switch",
    "contact_threshold_mutation",
    "program_patch",
    "crossover",
    "depth_refinement",
]


@dataclass
class OperatorStats:
    attempts: int = 0
    accepted: int = 0
    official_evaluated: int = 0
    valid_count: int = 0
    best_delta: float = 0.0
    mean_delta: float = 0.0
    avg_runtime: float = 0.0
    novelty_mean: float = 0.0
    common_failure_types: Dict[str, int] = field(default_factory=dict)


class StrategyBandit:
    def __init__(self):
        self.stats: Dict[str, OperatorStats] = {name: OperatorStats() for name in OPERATORS}
        self._delta_totals: Dict[str, float] = {name: 0.0 for name in OPERATORS}
        self._runtime_totals: Dict[str, float] = {name: 0.0 for name in OPERATORS}
        self._novelty_totals: Dict[str, float] = {name: 0.0 for name in OPERATORS}
        self._failures: Dict[str, Counter] = {name: Counter() for name in OPERATORS}

    def select(self, generation: int, task: str) -> str:
        total = sum(stat.attempts for stat in self.stats.values()) + 1
        best_name = OPERATORS[generation % len(OPERATORS)]
        best_score = -1e18
        for name in OPERATORS:
            stat = self.stats[name]
            if stat.attempts == 0:
                score = 1.0 + 0.03 * ((generation + len(task)) % len(OPERATORS))
            else:
                validity_bonus = 0.15 * (stat.valid_count / max(1, stat.official_evaluated))
                novelty_bonus = 0.12 * stat.novelty_mean
                runtime_penalty = min(0.2, stat.avg_runtime / 30.0)
                repeated_failure_penalty = 0.03 * max(0, stat.common_failure_types.get("rejected_novelty", 0) - 2)
                exploration_bonus = 0.2 * math.sqrt(math.log(total + 1) / stat.attempts)
                score = stat.mean_delta + novelty_bonus + validity_bonus - runtime_penalty - repeated_failure_penalty + exploration_bonus
            if score > best_score:
                best_name = name
                best_score = score
        return best_name

    def record_attempt(self, operator: str, accepted: bool, official_evaluated: bool,
                       valid: bool, delta: float, runtime: float, novelty_score: float,
                       failure_type: str = "none") -> None:
        if operator not in self.stats:
            self.stats[operator] = OperatorStats()
            self._delta_totals[operator] = 0.0
            self._runtime_totals[operator] = 0.0
            self._novelty_totals[operator] = 0.0
            self._failures[operator] = Counter()
        stat = self.stats[operator]
        stat.attempts += 1
        if accepted:
            stat.accepted += 1
        if official_evaluated:
            stat.official_evaluated += 1
        if valid:
            stat.valid_count += 1
        self._delta_totals[operator] += float(delta)
        self._runtime_totals[operator] += float(runtime)
        self._novelty_totals[operator] += float(novelty_score)
        stat.mean_delta = self._delta_totals[operator] / max(1, stat.attempts)
        stat.avg_runtime = self._runtime_totals[operator] / max(1, stat.attempts)
        stat.novelty_mean = self._novelty_totals[operator] / max(1, stat.attempts)
        stat.best_delta = max(float(stat.best_delta), float(delta))
        self._failures[operator][failure_type or "none"] += 1
        stat.common_failure_types = dict(self._failures[operator].most_common(5))

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: asdict(stat) for name, stat in sorted(self.stats.items())}

    def write(self, repo_root: Path) -> Path:
        path = Path(repo_root) / "agent" / "archive" / "evolve" / "operator_stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

