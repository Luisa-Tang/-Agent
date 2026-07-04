"""Strategy portfolio controller for Agent candidate selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from geometry_utils import TASK_SPECS


@dataclass
class StrategyDecision:
    strategy: str
    decision_reason: str
    portfolio_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    factors: Dict[str, Any] = field(default_factory=dict)


class StrategyPortfolioController:
    """Choose strategies using archive history, evaluator feedback, and budget."""

    BENCHMARK_SEED = "benchmark_seed_dominikkamp"
    REFINEMENT = {
        "fixed_centers_radius_lp",
        "micro_perturb_lp_refine",
        "optional_fico_task_a_seed",
    }

    def stats(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        deltas: Dict[str, list] = {}
        runtimes: Dict[str, list] = {}
        failures: Dict[str, Counter] = {}
        for record in records:
            strategy = str(record.get("strategy") or "unknown")
            item = grouped.setdefault(
                strategy,
                {
                    "attempts": 0,
                    "valid_count": 0,
                    "validity_rate": 0.0,
                    "best_score": 0.0,
                    "avg_score_delta": 0.0,
                    "avg_runtime": 0.0,
                    "common_failure_types": {},
                    "last_used_iteration": None,
                    "expected_improvement": 0.0,
                    "novelty_bonus": 0.0,
                    "runtime_penalty": 0.0,
                    "plateau_penalty": 0.0,
                    "repeated_code_penalty": 0.0,
                    "ucb_score": 0.0,
                },
            )
            deltas.setdefault(strategy, [])
            runtimes.setdefault(strategy, [])
            failures.setdefault(strategy, Counter())

            item["attempts"] += 1
            if record.get("valid"):
                item["valid_count"] += 1
            item["best_score"] = max(float(item["best_score"]), float(record.get("score") or 0.0))
            deltas[strategy].append(float(record.get("score_improvement") or 0.0))
            runtime = _parse_elapsed(record)
            if runtime is not None:
                runtimes[strategy].append(runtime)
            failures[strategy][str(record.get("failure_type") or "none")] += 1
            item["last_used_iteration"] = int(record.get("iteration") or 0)

        for strategy, item in grouped.items():
            attempts = max(1, int(item["attempts"]))
            item["validity_rate"] = float(item["valid_count"]) / attempts
            item["avg_score_delta"] = sum(deltas[strategy]) / max(1, len(deltas[strategy]))
            item["avg_runtime"] = sum(runtimes[strategy]) / max(1, len(runtimes[strategy])) if runtimes[strategy] else 0.0
            item["common_failure_types"] = dict(failures[strategy].most_common(4))
            item["expected_improvement"] = max(0.0, float(item["avg_score_delta"]))
            item["novelty_bonus"] = _novelty_bonus(strategy, attempts)
            item["runtime_penalty"] = min(0.2, float(item["avg_runtime"]) / 60.0)
            item["plateau_penalty"] = 0.05 if item["common_failure_types"].get("plateau") else 0.0
            item["repeated_code_penalty"] = _repeated_code_penalty(strategy, attempts)
            item["ucb_score"] = (
                item["expected_improvement"]
                + 0.25 * item["validity_rate"]
                + item["novelty_bonus"]
                - item["runtime_penalty"]
                - item["plateau_penalty"]
                - item["repeated_code_penalty"]
            )
        return grouped

    def decide(self, task: str, iteration: int, records: Iterable[Dict[str, Any]],
               best_record: Optional[Dict[str, Any]], last_record: Optional[Dict[str, Any]],
               no_improve: int, budget: Dict[str, Any],
               use_benchmark_seeds: bool = False,
               refine_benchmark: bool = False) -> StrategyDecision:
        records = list(records)
        stats = self.stats(records)
        counts = Counter(str(record.get("strategy") or "unknown") for record in records)
        target = float(TASK_SPECS[task]["target"])
        best_score = float(best_record.get("score") or 0.0) if best_record else 0.0
        best_gap = max(0.0, 1.0 - best_score)
        factors = {
            "no_valid_candidate": best_record is None,
            "last_failure": (last_record or {}).get("failure_type"),
            "plateau": no_improve >= 2,
            "best_score_gap_to_target_ratio": best_gap,
            "remaining_iterations": budget.get("remaining_iterations"),
            "remaining_seconds": budget.get("remaining_seconds"),
            "target_denominator": target,
        }

        if use_benchmark_seeds and counts[self.BENCHMARK_SEED] == 0:
            return StrategyDecision(
                self.BENCHMARK_SEED,
                "Public benchmark seed is enabled and has not yet been evaluated; use it as the first warm-start candidate.",
                stats,
                factors,
            )

        if best_record is None:
            strategy = self._prefer_successful(stats, ["baseline_safe_grid", "hexagonal_or_staggered_initialization", "scipy_slsqp_joint"])
            return StrategyDecision(strategy, "No valid candidate exists yet; select the most reliable valid-start strategy.", stats, factors)

        if last_record and not last_record.get("valid"):
            failure = str(last_record.get("failure_type") or "unknown")
            if failure in {"overlap", "boundary_violation", "nonfinite", "shape_error", "perimeter_error", "negative_radius"}:
                return StrategyDecision(
                    "hexagonal_or_staggered_initialization",
                    f"Official evaluator failure `{failure}` calls for a conservative structured repair path.",
                    stats,
                    factors,
                )

        if refine_benchmark:
            if counts["fixed_centers_radius_lp"] == 0:
                return StrategyDecision("fixed_centers_radius_lp", "Benchmark refinement is enabled; first try radii-only LP on current best centers.", stats, factors)
            if counts["micro_perturb_lp_refine"] == 0:
                return StrategyDecision("micro_perturb_lp_refine", "Radii-only LP has been tried; next probe tiny benchmark-neighborhood perturbations.", stats, factors)
            if task == "A" and counts["optional_fico_task_a_seed"] == 0:
                return StrategyDecision("optional_fico_task_a_seed", "Task A can optionally try a public FICO Problem 13 seed if a local copy exists.", stats, factors)

        if no_improve >= 2:
            options = ["perturb_best_and_repair", "multi_start_slsqp"]
            if refine_benchmark:
                options.insert(0, "micro_perturb_lp_refine")
            strategy = self._prefer_successful(stats, options)
            return StrategyDecision(strategy, "Plateau detected; choose a historically stronger exploit/diversify strategy.", stats, factors)

        if best_gap > 0.02:
            strategy = self._prefer_successful(stats, ["multi_start_slsqp", "scipy_slsqp_joint", "hexagonal_or_staggered_initialization"])
            return StrategyDecision(strategy, "Best score remains far from the target band; prioritize optimization strategies with good history.", stats, factors)

        cycle = ["hexagonal_or_staggered_initialization", "scipy_slsqp_joint", "multi_start_slsqp", "perturb_best_and_repair"]
        strategy = self._least_recent_successful(stats, cycle, iteration)
        return StrategyDecision(strategy, "Near-target search: balance historical success with recency to keep the portfolio diverse.", stats, factors)

    def _prefer_successful(self, stats: Dict[str, Dict[str, Any]], options: list) -> str:
        def key(strategy: str):
            stat = stats.get(strategy) or {}
            return (
                float(stat.get("ucb_score") or 0.0),
                float(stat.get("validity_rate") or 0.0),
                float(stat.get("expected_improvement") or 0.0),
                -float(stat.get("avg_runtime") or 0.0),
            )

        return max(options, key=key)

    def _least_recent_successful(self, stats: Dict[str, Dict[str, Any]], options: list, iteration: int) -> str:
        def key(strategy: str):
            stat = stats.get(strategy) or {}
            last = stat.get("last_used_iteration")
            age = iteration + 1 if last is None else iteration - int(last)
            return (
                age,
                float(stat.get("validity_rate") or 0.0),
                float(stat.get("best_score") or 0.0),
            )

        return max(options, key=key)


def _parse_elapsed(record: Dict[str, Any]) -> Optional[float]:
    trace = record.get("trace") or {}
    next_observation = trace.get("next_observation") or {}
    value = next_observation.get("elapsed") or record.get("elapsed")
    if value is None:
        return None
    text = str(value).strip().lower().replace("s", "")
    try:
        return float(text)
    except ValueError:
        return None


def _novelty_bonus(strategy: str, attempts: int) -> float:
    if attempts == 0:
        return 0.15
    if strategy in {"contact_graph_feasibility_refine", "public_frontier_dominikkamp", "public_frontier_fico_task_a"}:
        return 0.08 / (1.0 + 0.25 * attempts)
    return 0.03 / (1.0 + attempts)


def _repeated_code_penalty(strategy: str, attempts: int) -> float:
    if attempts <= 2:
        return 0.0
    if strategy in {"perturb_best_and_repair", "micro_perturb_lp_refine"}:
        return min(0.12, 0.01 * (attempts - 2))
    return min(0.06, 0.005 * (attempts - 2))
