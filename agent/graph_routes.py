"""Conditional routing for the optional LangGraph orchestration layer."""

from __future__ import annotations

from typing import Any, Dict


BENCHMARK_SEED_STRATEGY = "benchmark_seed_dominikkamp"
REPAIR_STRATEGY = "hexagonal_or_staggered_initialization"
EXPLORE_STRATEGY = "multi_start_slsqp"
EXPLOIT_STRATEGY = "perturb_best_and_repair"
REPAIR_FAILURES = {"overlap", "boundary_violation", "nonfinite", "shape_error", "perimeter_error", "negative_radius"}


def update_next_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """Set next_action and, when useful, the next selected strategy."""
    updated = dict(state)
    artifacts = dict(updated.get("artifacts") or {})
    used_strategies = set(artifacts.get("used_strategies") or [])
    iteration = int(updated.get("iteration") or 0)
    max_iterations = int(updated.get("max_iterations") or 0)
    failure_type = str(updated.get("failure_type") or "none")
    best_score = float(updated.get("best_score") or 0.0)
    eval_result = updated.get("eval_result") or {}
    valid_improvement = bool(eval_result.get("valid")) and float(eval_result.get("score_improvement") or 0.0) > 1e-12

    if iteration >= max_iterations:
        updated["next_action"] = "static_export"
        updated["decision_reason"] = "Iteration budget reached; export the best official-evaluator-valid candidate."
        return updated

    if failure_type in REPAIR_FAILURES:
        updated["selected_strategy"] = REPAIR_STRATEGY
        updated["next_action"] = "generate_candidate"
        updated["decision_reason"] = (
            f"Official evaluator failure `{failure_type}` routes directly to the structured repair strategy."
        )
        return updated

    benchmark_allowed = bool(artifacts.get("benchmark_allowed"))
    if benchmark_allowed and best_score < 0.999 and BENCHMARK_SEED_STRATEGY not in used_strategies:
        updated["selected_strategy"] = BENCHMARK_SEED_STRATEGY
        updated["next_action"] = "generate_candidate"
        updated["decision_reason"] = (
            "Best score is still below 0.999 and the public benchmark seed has not been tried; route to benchmark warm-start."
        )
        return updated

    if bool(artifacts.get("plateau")):
        updated["selected_strategy"] = EXPLORE_STRATEGY
        updated["next_action"] = "generate_candidate"
        updated["decision_reason"] = "Plateau detected; route to an exploration strategy from the portfolio."
        return updated

    if valid_improvement:
        updated["selected_strategy"] = EXPLOIT_STRATEGY
        updated["next_action"] = "generate_candidate"
        updated["decision_reason"] = "Valid improvement archived; route to exploit the current best geometry."
        return updated

    updated["next_action"] = "select_strategy"
    updated["decision_reason"] = "No special route fired; return to portfolio-based strategy selection."
    return updated


def route_next(state: Dict[str, Any]) -> str:
    action = str(state.get("next_action") or "select_strategy")
    if action in {"select_strategy", "generate_candidate", "static_export", "safety_check"}:
        return action
    if action in {"end", "END"}:
        return "end"
    return "select_strategy"

