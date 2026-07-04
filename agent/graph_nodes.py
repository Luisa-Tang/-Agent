"""LangGraph node functions that reuse the existing GeoOpt Agent modules."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from archive import ArchiveManager, sanitize_json
from candidate_generators import CandidateGenerator
from evaluator_adapter import EvaluatorAdapter, EvalResult
from geometry_utils import PackingData, safety_metrics
from graph_routes import update_next_action
from lineage import hash_packing_data, hash_text
from log_utils import utc_stamp
from safety_guard import SafetyGuard
from state import GeoOptState
from strategy_controller import StrategyPortfolioController


BENCHMARK_SEED_STRATEGY = "benchmark_seed_dominikkamp"


@dataclass
class GraphRuntime:
    repo_root: Path
    run_id: str
    seed: int
    max_iterations: int
    fast: bool
    use_benchmark_seeds: bool
    archive: ArchiveManager
    adapter: EvaluatorAdapter
    generator: CandidateGenerator
    controller: StrategyPortfolioController
    safety_guard: SafetyGuard
    task: str
    graph_run_log: Path
    metrics_graph_run_log: Path
    best_data: Optional[PackingData] = None
    current_candidate: Any = None
    current_eval: Optional[EvalResult] = None
    last_record: Optional[Dict[str, Any]] = None
    no_improve: int = 0


def load_task(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    task_dir = runtime.adapter.task_dir(runtime.task)
    loaded = []
    for name in ("task_description.md", "evaluate.py", "baseline.py"):
        path = task_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")
        _ = path.read_text(encoding="utf-8")
        loaded.append(_rel(runtime.repo_root, path))
    artifacts = _artifacts(updated)
    artifacts["loaded_task_files"] = loaded
    artifacts["benchmark_allowed"] = bool(runtime.use_benchmark_seeds)
    updated.update(
        {
            "task": runtime.task,
            "iteration": int(updated.get("iteration") or 0),
            "max_iterations": int(updated.get("max_iterations") or runtime.max_iterations),
            "seed": int(updated.get("seed") or runtime.seed),
            "artifacts": artifacts,
            "next_action": "observe_archive",
        }
    )
    _log_node(runtime, "load_task", before, _state_summary(updated), updated)
    return updated


def observe_archive(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    records = runtime.archive.records_for_task(runtime.task)
    best = runtime.archive.best_record(runtime.task)
    stats = runtime.controller.stats(records)
    artifacts = _artifacts(updated)
    artifacts["used_strategies"] = sorted({str(record.get("strategy") or "unknown") for record in records})
    artifacts["plateau"] = runtime.no_improve >= 2
    artifacts["records_for_task"] = len(records)
    updated.update(
        {
            "archive_summary": {
                "records_for_task": len(records),
                "best_candidate": _compact_record(best),
                "no_improve_count": runtime.no_improve,
            },
            "strategy_stats": stats,
            "best_candidate_id": best.get("candidate_id") if best else None,
            "best_score": float(best.get("score") or 0.0) if best else 0.0,
            "best_sum_radii": float(best.get("sum_radii") or 0.0) if best else 0.0,
            "artifacts": artifacts,
            "next_action": "select_strategy",
        }
    )
    _log_node(runtime, "observe_archive", before, _state_summary(updated), updated)
    return updated


def select_strategy(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    iteration = int(updated.get("iteration") or 0)
    budget = {
        "iteration_limit": int(updated.get("max_iterations") or runtime.max_iterations),
        "remaining_iterations": max(0, int(updated.get("max_iterations") or runtime.max_iterations) - iteration - 1),
        "fast": bool(runtime.fast),
    }
    decision = runtime.controller.decide(
        task=runtime.task,
        iteration=iteration,
        records=runtime.archive.records_for_task(runtime.task),
        best_record=runtime.archive.best_record(runtime.task),
        last_record=runtime.last_record,
        no_improve=runtime.no_improve,
        budget=budget,
        use_benchmark_seeds=runtime.use_benchmark_seeds,
        refine_benchmark=False,
    )
    artifacts = _artifacts(updated)
    artifacts["portfolio_factors"] = decision.factors
    artifacts["portfolio_stats"] = decision.portfolio_stats
    updated.update(
        {
            "selected_strategy": decision.strategy,
            "decision_reason": decision.decision_reason,
            "skills_used": _skills_for_strategy(decision.strategy),
            "artifacts": artifacts,
            "next_action": "generate_candidate",
        }
    )
    _log_node(runtime, "select_strategy", before, _state_summary(updated), updated)
    return updated


def generate_candidate(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    strategy = str(updated.get("selected_strategy") or "baseline_safe_grid")
    iteration = int(updated.get("iteration") or 0)
    parent_record = runtime.archive.best_record(runtime.task)
    candidate_id = runtime.archive.make_candidate_id(runtime.task, iteration, strategy)
    candidate = runtime.generator.generate(
        runtime.task,
        strategy,
        iteration=iteration,
        parent_data=runtime.best_data,
        feedback=runtime.last_record,
    )
    runtime.current_candidate = candidate
    runtime.current_eval = None
    metrics = safety_metrics(candidate.data)
    source_metadata = candidate.diagnostics.get("source_metadata")
    if isinstance(source_metadata, dict):
        source_metadata = dict(source_metadata)
    else:
        source_metadata = None
    artifacts = _artifacts(updated)
    artifacts.update(
        {
            "candidate_strategy": strategy,
            "generated_code_bytes": len(candidate.code.encode("utf-8")),
            "geometry_metrics": metrics,
            "source_metadata": source_metadata,
            "code_hash": hash_text(candidate.code),
            "data_hash": hash_packing_data(candidate.data),
        }
    )
    updated.update(
        {
            "candidate_id": candidate_id,
            "parent_candidate_id": parent_record.get("candidate_id") if parent_record else None,
            "selected_strategy": strategy,
            "skills_used": _skills_for_strategy(strategy),
            "artifacts": artifacts,
            "next_action": "evaluate_candidate",
        }
    )
    _log_node(runtime, "generate_candidate", before, _state_summary(updated), updated)
    return updated


def evaluate_candidate(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    if runtime.current_candidate is None:
        raise RuntimeError("evaluate_candidate called before generate_candidate")
    result = runtime.adapter.evaluate_task(runtime.task, runtime.current_candidate.code)
    runtime.current_eval = result
    eval_payload = _eval_payload(result)
    artifacts = _artifacts(updated)
    artifacts["raw_evaluator_output_preview"] = result.raw_output[:1200]
    updated.update(
        {
            "eval_result": eval_payload,
            "failure_type": result.failure_type,
            "artifacts": artifacts,
            "next_action": "parse_feedback",
        }
    )
    _log_node(runtime, "evaluate_candidate", before, _state_summary(updated), updated)
    return updated


def parse_feedback(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    eval_result = dict(updated.get("eval_result") or {})
    best_score = float(updated.get("best_score") or 0.0)
    score = float(eval_result.get("score") or 0.0)
    score_delta = score - best_score
    eval_result["score_improvement"] = score_delta
    failure_type = _effective_failure_type(eval_result, score_delta, runtime.no_improve)
    artifacts = _artifacts(updated)
    artifacts["feedback_classification"] = failure_type
    updated.update(
        {
            "eval_result": eval_result,
            "failure_type": failure_type,
            "artifacts": artifacts,
            "next_action": "update_archive",
        }
    )
    _log_node(runtime, "parse_feedback", before, _state_summary(updated), updated)
    return updated


def update_archive(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    if runtime.current_candidate is None or runtime.current_eval is None:
        raise RuntimeError("update_archive requires a generated and evaluated candidate")

    candidate = runtime.current_candidate
    eval_result = runtime.current_eval
    candidate_id = str(updated.get("candidate_id") or runtime.archive.make_candidate_id(
        runtime.task, int(updated.get("iteration") or 0), str(updated.get("selected_strategy") or "unknown")
    ))
    code_snapshot = runtime.archive.save_candidate_code(runtime.task, candidate_id, candidate.code)
    raw_output = runtime.archive.save_raw_output(runtime.task, candidate_id, eval_result.raw_output)
    previous_best = runtime.archive.best_record(runtime.task)
    best_score = float(previous_best.get("score") or 0.0) if previous_best else 0.0
    score_delta = float(eval_result.score) - best_score
    failure_type = _effective_failure_type(_eval_payload(eval_result), score_delta, runtime.no_improve)
    metrics = safety_metrics(candidate.data)
    artifacts = _artifacts(updated)
    source_metadata = artifacts.get("source_metadata")
    if isinstance(source_metadata, dict):
        source_metadata = dict(source_metadata)
        source_metadata["official_evaluator_result"] = {
            "valid": eval_result.valid,
            "score": eval_result.score,
            "sum_radii": eval_result.sum_radii,
            "failure_type": eval_result.failure_type,
            "returncode": eval_result.returncode,
            "elapsed": eval_result.elapsed_text,
            "raw_output_path": str(raw_output),
            "exact_sum_radii": eval_result.exact_sum_radii,
            "exact_score": eval_result.exact_score,
        }
        candidate.diagnostics["source_metadata"] = source_metadata

    record = {
        "task": runtime.task,
        "iteration": int(updated.get("iteration") or 0),
        "candidate_id": candidate_id,
        "strategy": str(updated.get("selected_strategy") or candidate.strategy),
        "parent_candidate_id": updated.get("parent_candidate_id"),
        "valid": eval_result.valid,
        "score": eval_result.score,
        "sum_radii": eval_result.sum_radii if eval_result.sum_radii is not None else candidate.data.sum_radii,
        "width": candidate.data.width if runtime.task == "A" else None,
        "height": candidate.data.height if runtime.task == "A" else None,
        "failure_type": failure_type,
        "raw_failure_type": eval_result.failure_type,
        "score_improvement": score_delta,
        "decision": _decision_after_result(eval_result.valid, failure_type, eval_result.score, previous_best),
        "decision_reason": updated.get("decision_reason") or "",
        "raw_output": raw_output,
        "code_snapshot": code_snapshot,
        "code_hash": artifacts.get("code_hash") or hash_text(candidate.code),
        "data_hash": artifacts.get("data_hash") or hash_packing_data(candidate.data),
        "input_artifacts": {
            "parent_candidate_id": updated.get("parent_candidate_id"),
            "source_metadata": source_metadata,
        },
        "output_artifacts": {
            "code_snapshot": str(code_snapshot),
            "raw_output": str(raw_output),
            "solution_path": str(runtime.adapter.solution_path(runtime.task)),
        },
        "diagnostics": candidate.diagnostics,
        "geometry_metrics": metrics,
        "source_metadata": source_metadata,
        "skills_used": updated.get("skills_used") or _skills_for_strategy(str(updated.get("selected_strategy") or "")),
        "trace": {
            "graph_node": "update_archive",
            "selected_strategy": updated.get("selected_strategy"),
            "eval_result": _eval_payload(eval_result),
            "next_action_before_route": updated.get("next_action"),
        },
    }
    runtime.archive.add_record(record)
    current_best = runtime.archive.best_record(runtime.task)
    improved = bool(current_best and current_best.get("candidate_id") == candidate_id)
    if improved:
        runtime.best_data = candidate.data
        runtime.no_improve = 0
    else:
        runtime.no_improve += 1
    runtime.last_record = record

    artifacts.update(
        {
            "code_snapshot": str(code_snapshot),
            "raw_output": str(raw_output),
            "geometry_metrics": metrics,
            "source_metadata": source_metadata,
            "used_strategies": sorted({str(item.get("strategy") or "unknown") for item in runtime.archive.records_for_task(runtime.task)}),
            "plateau": runtime.no_improve >= 2,
        }
    )
    eval_payload = _eval_payload(eval_result)
    eval_payload["score_improvement"] = score_delta
    next_iteration = int(updated.get("iteration") or 0) + 1
    updated.update(
        {
            "iteration": next_iteration,
            "eval_result": eval_payload,
            "failure_type": failure_type,
            "best_candidate_id": current_best.get("candidate_id") if current_best else None,
            "best_score": float(current_best.get("score") or 0.0) if current_best else 0.0,
            "best_sum_radii": float(current_best.get("sum_radii") or 0.0) if current_best else 0.0,
            "archive_summary": {
                "records_for_task": len(runtime.archive.records_for_task(runtime.task)),
                "best_candidate": _compact_record(current_best),
                "no_improve_count": runtime.no_improve,
            },
            "strategy_stats": runtime.controller.stats(runtime.archive.records_for_task(runtime.task)),
            "artifacts": artifacts,
        }
    )
    updated = update_next_action(updated)
    _log_node(runtime, "update_archive", before, _state_summary(updated), updated)
    return updated


def static_export(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    exported = runtime.archive.export_best_code(runtime.task, runtime.adapter.solution_path(runtime.task))
    if exported is None:
        fallback = runtime.generator.generate(runtime.task, "baseline_safe_grid", iteration=999)
        runtime.adapter.write_candidate(runtime.task, fallback.code)
        exported = runtime.adapter.solution_path(runtime.task)
    final_eval = runtime.adapter.evaluate_task(runtime.task)
    artifacts = _artifacts(updated)
    artifacts["static_export_path"] = str(exported)
    artifacts["static_export_eval"] = _eval_payload(final_eval)
    updated.update(
        {
            "eval_result": _eval_payload(final_eval),
            "failure_type": final_eval.failure_type,
            "artifacts": artifacts,
            "next_action": "safety_check",
        }
    )
    _log_node(runtime, "static_export", before, _state_summary(updated), updated)
    return updated


def safety_check(state: GeoOptState, runtime: GraphRuntime) -> GeoOptState:
    before = _state_summary(state)
    updated = dict(state)
    report = runtime.safety_guard.check_post_run(runtime.run_id, write_submission=True)
    artifacts = _artifacts(updated)
    artifacts["safety_report"] = {
        "path": "submission/safety_report.json",
        "passed": report.get("passed"),
        "protected_files_unchanged": (report.get("protected_files") or {}).get("unchanged"),
    }
    updated.update({"artifacts": artifacts, "next_action": "end"})
    _log_node(runtime, "safety_check", before, _state_summary(updated), updated)
    return updated


def _artifacts(state: Dict[str, Any]) -> Dict[str, Any]:
    return dict(state.get("artifacts") or {})


def _eval_payload(result: EvalResult) -> Dict[str, Any]:
    return {
        "valid": result.valid,
        "score": result.score,
        "sum_radii": result.sum_radii,
        "failure_type": result.failure_type,
        "returncode": result.returncode,
        "elapsed": result.elapsed_text,
        "exact_sum_radii": result.exact_sum_radii,
        "exact_score": result.exact_score,
        "exact_width": result.exact_width,
        "exact_height": result.exact_height,
    }


def _effective_failure_type(eval_result: Dict[str, Any], score_improvement: float, no_improve: int) -> str:
    if not eval_result.get("valid"):
        return str(eval_result.get("failure_type") or "unknown")
    if no_improve >= 2 and score_improvement <= 1e-9:
        return "plateau"
    if float(eval_result.get("score") or 0.0) < 0.90:
        return "low_score"
    return "none"


def _decision_after_result(valid: bool, failure_type: str, score: float,
                           previous_best: Optional[Dict[str, Any]]) -> str:
    if not valid:
        return f"Official evaluator rejected candidate with failure `{failure_type}`."
    best_score = float(previous_best.get("score") or 0.0) if previous_best else 0.0
    if float(score) > best_score:
        return "Official evaluator accepted an improved candidate; archive as current best."
    return "Official evaluator accepted candidate, but it did not improve the archive best."


def _skills_for_strategy(strategy: str) -> list:
    skills = ["archive-observability"]
    if strategy in {"scipy_slsqp_joint", "multi_start_slsqp"}:
        skills.append("packing-slsqp")
    if strategy in {
        "perturb_best_and_repair",
        "baseline_safe_grid",
        "hexagonal_or_staggered_initialization",
        "fixed_centers_radius_lp",
        "micro_perturb_lp_refine",
        "optional_fico_task_a_seed",
    }:
        skills.append("packing-repair")
    if strategy == BENCHMARK_SEED_STRATEGY:
        skills.append("static-export")
    skills.append("evaluator-feedback")
    return skills


def _compact_record(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    keys = [
        "candidate_id",
        "strategy",
        "valid",
        "score",
        "sum_radii",
        "width",
        "height",
        "failure_type",
        "score_improvement",
    ]
    return {key: record.get(key) for key in keys if key in record}


def _state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = state.get("artifacts") or {}
    return {
        "task": state.get("task"),
        "iteration": state.get("iteration"),
        "max_iterations": state.get("max_iterations"),
        "selected_strategy": state.get("selected_strategy"),
        "decision_reason": state.get("decision_reason"),
        "candidate_id": state.get("candidate_id"),
        "parent_candidate_id": state.get("parent_candidate_id"),
        "failure_type": state.get("failure_type"),
        "best_candidate_id": state.get("best_candidate_id"),
        "best_score": state.get("best_score"),
        "best_sum_radii": state.get("best_sum_radii"),
        "next_action": state.get("next_action"),
        "skills_used": state.get("skills_used") or [],
        "artifact_keys": sorted(artifacts.keys()),
    }


def _log_node(runtime: GraphRuntime, graph_node: str, before: Dict[str, Any],
              after: Dict[str, Any], state: Dict[str, Any]) -> None:
    record = {
        "timestamp_utc": utc_stamp(),
        "run_id": runtime.run_id,
        "task": runtime.task,
        "graph_node": graph_node,
        "state_before_summary": before,
        "state_after_summary": after,
        "selected_strategy": state.get("selected_strategy"),
        "eval_result": state.get("eval_result"),
        "next_action": state.get("next_action"),
        "artifacts": state.get("artifacts") or {},
        "elapsed_marker": time.time(),
    }
    line = json.dumps(sanitize_json(record), sort_keys=True)
    for path in (runtime.graph_run_log, runtime.metrics_graph_run_log):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)

