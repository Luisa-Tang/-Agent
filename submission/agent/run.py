"""Main entrypoint for the local circle-packing Agent."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from archive import ArchiveManager
from batch_candidate_runner import BreakthroughConfig, run_breakthrough_search
from candidate_generators import CandidateGenerator
from evaluator_adapter import EvaluatorAdapter
from evolve.evolve_runner import SelfEvolveConfig, run_self_evolution_search
from geometry_utils import PackingData, safety_metrics
from lineage import hash_packing_data, hash_text, write_best_lineages
from llm_reflector import LLMReflector
from log_utils import append_final_summary, append_human_iteration, reset_human_log
from report_data import generate_report
from safety_guard import SafetyGuard
from state import AgentState
from strategy_controller import StrategyPortfolioController


BENCHMARK_SEED_STRATEGY = "benchmark_seed_dominikkamp"
REFINEMENT_STRATEGIES = {
    "fixed_centers_radius_lp",
    "micro_perturb_lp_refine",
    "optional_fico_task_a_seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Agent for AlgorithmOptimization circle packing")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Use fewer starts and lower optimizer iteration budgets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-benchmark-seeds", action="store_true",
                        help="Allow public external benchmark warm-start candidates")
    parser.add_argument("--refine-benchmark", action="store_true",
                        help="Run lightweight benchmark-neighborhood LP refinements")
    parser.add_argument("--breakthrough-search", action="store_true",
                        help="Run optional score breakthrough harness after the stable agent loop")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for breakthrough candidate generation")
    parser.add_argument("--max-breakthrough-candidates", type=int, default=200,
                        help="Maximum generated breakthrough candidates across selected tasks")
    parser.add_argument("--target-score-a", type=float, default=1.0)
    parser.add_argument("--target-score-b", type=float, default=1.0)
    parser.add_argument("--self-evolve-search", action="store_true",
                        help="Run optional GeoEvolve-lite program self-evolution after the stable loop")
    parser.add_argument("--evolve-generations", type=int, default=20)
    parser.add_argument("--evolve-batch-size", type=int, default=4)
    parser.add_argument("--max-official-evals", type=int, default=40)
    parser.add_argument("--novelty-threshold", type=float, default=0.25)
    parser.add_argument("--time-limit", type=int, default=None, help="Optional whole-run wall clock limit in seconds")
    parser.add_argument("--use-llm", action="store_true", help="Enable optional LLM strategy reflection")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--llm-model", default="deepseek-v4-pro")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive = ArchiveManager(REPO_ROOT, run_id)
    adapter = EvaluatorAdapter(REPO_ROOT, python_executable=sys.executable)
    generator = CandidateGenerator(seed=args.seed, fast=args.fast)
    controller = StrategyPortfolioController()
    safety_guard = SafetyGuard(REPO_ROOT)
    safety_guard.capture_pre_run()
    reset_skill_usage(REPO_ROOT, run_id)
    reflector = LLMReflector(
        enabled=args.use_llm,
        repo_root=REPO_ROOT,
        model=args.llm_model,
        base_url=args.llm_base_url,
    )
    tasks = ["A", "B"] if args.task == "both" else [args.task]

    print(f"repo_root={REPO_ROOT}")
    print(f"run_id={run_id}")
    print(f"python={sys.executable}")
    print(
        f"tasks={tasks}, iterations={args.iterations}, fast={args.fast}, seed={args.seed}, "
        f"use_benchmark_seeds={args.use_benchmark_seeds}, "
        f"refine_benchmark={args.refine_benchmark}, "
        f"breakthrough_search={args.breakthrough_search}, "
        f"self_evolve_search={args.self_evolve_search}, "
        f"use_llm={args.use_llm}, llm_model={args.llm_model}"
    )

    start = time.time()
    for task in tasks:
        read_task_context(task)
        run_task_loop(task, args, generator, adapter, archive, reflector, controller, start)

    if args.breakthrough_search:
        breakthrough_summary = run_breakthrough_search(
            REPO_ROOT,
            tasks,
            archive,
            adapter,
            BreakthroughConfig(
                batch_size=max(1, int(args.batch_size)),
                max_candidates=max(0, int(args.max_breakthrough_candidates)),
                seed=int(args.seed),
                use_benchmark_seeds=bool(args.use_benchmark_seeds),
                target_score_a=float(args.target_score_a),
                target_score_b=float(args.target_score_b),
            ),
        )
        print(f"breakthrough_summary={REPO_ROOT / 'agent' / 'archive' / 'metrics' / 'breakthrough_summary.json'}")
        for task, item in sorted((breakthrough_summary.get("tasks") or {}).items()):
            best = item.get("best_after") or {}
            print(
                f"Breakthrough Task {task}: score={float(best.get('score') or 0.0):.9f} "
                f"sum={float(best.get('sum_radii') or 0.0):.9f} "
                f"gap={float(item.get('gap_to_target') or 0.0):.9g} "
                f"exceeded={item.get('exceeded_target')}"
            )

    if args.self_evolve_search:
        self_evolve_summary = run_self_evolution_search(
            REPO_ROOT,
            tasks,
            archive,
            adapter,
            SelfEvolveConfig(
                generations=max(0, int(args.evolve_generations)),
                batch_size=max(1, int(args.evolve_batch_size)),
                max_official_evals=max(0, int(args.max_official_evals)),
                novelty_threshold=float(args.novelty_threshold),
                seed=int(args.seed),
                use_benchmark_seeds=bool(args.use_benchmark_seeds),
                use_llm=bool(args.use_llm),
            ),
        )
        print(f"self_evolution_summary={self_evolve_summary.get('summary_path')}")
        for task, item in sorted((self_evolve_summary.get("tasks") or {}).items()):
            after = item.get("best_after") or {}
            print(
                f"Self-evolve Task {task}: score={float(after.get('score') or 0.0):.9f} "
                f"sum={float(after.get('sum_radii') or 0.0):.9f} "
                f"gap={float(item.get('gap_to_denominator') or 0.0):.9g} "
                f"improved={item.get('improved_over_start')}"
            )

    summary_path = archive.write_summary()
    lineage_paths = write_best_lineages(REPO_ROOT, archive.records)
    write_strategy_portfolio_metrics(REPO_ROOT, controller.stats(archive.records), lineage_paths)
    evaluate_all_proc = adapter.run_evaluate_all(filename="solution.py")
    evaluate_all_output = evaluate_all_proc.stdout
    if evaluate_all_proc.stderr:
        evaluate_all_output += "\n[stderr]\n" + evaluate_all_proc.stderr
    print(evaluate_all_output)
    print(f"archive_summary={summary_path}")

    create_submission(REPO_ROOT, archive, evaluate_all_output, safety_guard)
    print(f"submission_dir={REPO_ROOT / 'submission'}")
    return 0 if evaluate_all_proc.returncode == 0 else evaluate_all_proc.returncode


def read_task_context(task: str) -> None:
    task_dir = REPO_ROOT / ("task_A" if task == "A" else "task_B")
    for name in ("task_description.md", "evaluate.py", "baseline.py"):
        path = task_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")
        _ = path.read_text(encoding="utf-8")


def run_task_loop(task: str, args: argparse.Namespace, generator: CandidateGenerator,
                  adapter: EvaluatorAdapter, archive: ArchiveManager,
                  reflector: LLMReflector,
                  controller: StrategyPortfolioController,
                  run_start: float) -> None:
    task_dir = adapter.task_dir(task)
    human_log = task_dir / ("run_log_a.log" if task == "A" else "run_log_b.log")
    reset_human_log(human_log, task, archive.run_id)

    best_data: Optional[PackingData] = None
    last_record: Optional[Dict] = None
    no_improve = 0

    for iteration in range(max(0, int(args.iterations))):
        if args.time_limit is not None and time.time() - run_start > args.time_limit:
            break
        previous_best = archive.best_record(task)
        observation = build_observation(task, iteration, previous_best, last_record, no_improve, archive)
        budget = {
            "iteration_limit": int(args.iterations),
            "remaining_iterations": max(0, int(args.iterations) - iteration - 1),
            "time_limit_seconds": args.time_limit,
            "elapsed_seconds": time.time() - run_start,
            "remaining_seconds": None if args.time_limit is None else max(0.0, float(args.time_limit) - (time.time() - run_start)),
            "fast": bool(args.fast),
        }
        state = AgentState(
            task,
            iteration,
            archive_summary={
                "best": compact_record(previous_best),
                "strategy_stats": archive.strategy_stats(task),
                "num_records_for_task": len(archive.records_for_task(task)),
                "no_improve_count": no_improve,
            },
            last_eval=compact_record(last_record),
            best_candidate_id=previous_best.get("candidate_id") if previous_best else None,
            budget=budget,
            next_action="decide",
        )
        state_flow = [state.snapshot("observe")]
        portfolio_decision = controller.decide(
            task=task,
            iteration=iteration,
            records=archive.records_for_task(task),
            best_record=previous_best,
            last_record=last_record,
            no_improve=no_improve,
            budget=budget,
            use_benchmark_seeds=args.use_benchmark_seeds,
            refine_benchmark=args.refine_benchmark,
        )
        local_strategy = portfolio_decision.strategy
        llm_decision = reflector.decide(
            task=task,
            iteration=iteration,
            best_record=previous_best,
            last_record=last_record,
            no_improve=no_improve,
            local_strategy=local_strategy,
            recent_records=archive.records_for_task(task),
        )
        strategy = local_strategy if local_strategy == BENCHMARK_SEED_STRATEGY else (llm_decision.strategy or local_strategy)
        decision_reason = strategy_reason(strategy, observation, llm_decision.reason)
        if strategy == local_strategy and (not llm_decision.used):
            decision_reason = portfolio_decision.decision_reason
        state.selected_strategy = strategy
        state.decision_reason = decision_reason
        state.next_action = "act"
        state.artifacts["portfolio_factors"] = portfolio_decision.factors
        state.artifacts["portfolio_stats"] = portfolio_decision.portfolio_stats
        state_flow.append(state.snapshot("decide"))
        thought = {
            "selected_strategy": strategy,
            "local_policy_strategy": local_strategy,
            "reason": decision_reason,
            "portfolio_decision": {
                "strategy": portfolio_decision.strategy,
                "decision_reason": portfolio_decision.decision_reason,
                "factors": portfolio_decision.factors,
            },
            "llm_decision": llm_decision.to_record(),
        }
        candidate = generator.generate(
            task,
            strategy,
            iteration=iteration,
            parent_data=best_data,
            feedback=last_record,
        )
        metrics = safety_metrics(candidate.data)
        code_hash = hash_text(candidate.code)
        data_hash = hash_packing_data(candidate.data)
        source_metadata = candidate.diagnostics.get("source_metadata")
        if isinstance(source_metadata, dict):
            source_metadata = dict(source_metadata)
        else:
            source_metadata = None
        skills_used = skills_for_iteration(strategy, eval_result=None)
        action = {
            "strategy": strategy,
            "optimizer_or_repair": action_summary(strategy),
            "skills_used": skills_used,
            "generated_code_bytes": len(candidate.code.encode("utf-8")),
            "geometry_metrics": metrics,
            "diagnostics": candidate.diagnostics,
            "source_metadata": source_metadata,
            "code_hash": code_hash,
            "data_hash": data_hash,
        }
        candidate_id = archive.make_candidate_id(task, iteration, strategy)
        state.skills_used = skills_used
        state.artifacts.update({
            "candidate_id": candidate_id,
            "code_hash": code_hash,
            "data_hash": data_hash,
            "generated_code_bytes": len(candidate.code.encode("utf-8")),
            "source_metadata": source_metadata,
        })
        state.next_action = "evaluate"
        state_flow.append(state.snapshot("act"))
        eval_result = adapter.evaluate_task(task, candidate.code)
        code_snapshot = archive.save_candidate_code(task, candidate_id, candidate.code)
        raw_output = archive.save_raw_output(task, candidate_id, eval_result.raw_output)

        if source_metadata is not None:
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
            candidate.diagnostics["official_evaluator_result"] = source_metadata["official_evaluator_result"]
            action["source_metadata"] = source_metadata

        score_improvement = float(eval_result.score) - float(previous_best.get("score") or 0.0) if previous_best else float(eval_result.score)
        effective_failure = effective_failure_type(eval_result, score_improvement, no_improve)
        skills_used = skills_for_iteration(strategy, eval_result=eval_result, effective_failure=effective_failure)
        action["skills_used"] = skills_used
        result_decision_reason = decision_after_result(eval_result.valid, effective_failure,
                                                       eval_result.score, previous_best)
        next_observation = {
            "valid": eval_result.valid,
            "score": eval_result.score,
            "sum_radii": eval_result.sum_radii if eval_result.sum_radii is not None else candidate.data.sum_radii,
            "failure_type": effective_failure,
            "raw_output": raw_output,
            "returncode": eval_result.returncode,
            "elapsed": eval_result.elapsed_text,
        }
        state.skills_used = skills_used
        state.last_eval = next_observation
        state.artifacts.update({
            "code_snapshot": str(code_snapshot),
            "raw_output": str(raw_output),
            "official_score": eval_result.score,
            "official_sum_radii": eval_result.sum_radii,
        })
        state.next_action = "archive"
        state_flow.append(state.snapshot("evaluate"))
        input_artifacts = {
            "parent_candidate_id": previous_best.get("candidate_id") if previous_best else None,
            "source_metadata": source_metadata,
        }
        output_artifacts = {
            "code_snapshot": str(code_snapshot),
            "raw_output": str(raw_output),
            "solution_path": str(adapter.solution_path(task)),
        }
        record = {
            "task": task,
            "iteration": iteration,
            "candidate_id": candidate_id,
            "strategy": strategy,
            "parent_candidate_id": previous_best.get("candidate_id") if previous_best else None,
            "valid": eval_result.valid,
            "score": eval_result.score,
            "sum_radii": eval_result.sum_radii if eval_result.sum_radii is not None else candidate.data.sum_radii,
            "width": candidate.data.width if task == "A" else None,
            "height": candidate.data.height if task == "A" else None,
            "failure_type": effective_failure,
            "raw_failure_type": eval_result.failure_type,
            "score_improvement": score_improvement,
            "decision": result_decision_reason,
            "decision_reason": result_decision_reason,
            "raw_output": raw_output,
            "code_snapshot": code_snapshot,
            "code_hash": code_hash,
            "data_hash": data_hash,
            "input_artifacts": input_artifacts,
            "output_artifacts": output_artifacts,
            "diagnostics": candidate.diagnostics,
            "geometry_metrics": metrics,
            "source_metadata": source_metadata,
            "local_policy_strategy": local_strategy,
            "llm_decision": llm_decision.to_record(),
            "skills_used": skills_used,
            "state": state.snapshot("archive_pending"),
            "state_flow": state_flow,
            "trace": {
                "observation": observation,
                "thought_decision": thought,
                "action": action,
                "next_observation": next_observation,
            },
        }
        state.best_candidate_id = candidate_id if eval_result.valid and score_improvement > 0.0 else state.best_candidate_id
        state.next_action = "next_iteration"
        state.artifacts["archive_decision"] = result_decision_reason
        record["state_flow"].append(state.snapshot("archive"))
        archive.add_record(record)
        current_best = archive.best_record(task)
        improved = bool(current_best and current_best.get("candidate_id") == candidate_id)
        if improved:
            best_data = candidate.data
            no_improve = 0
        else:
            no_improve += 1
        append_human_iteration(human_log, record, improved=improved)
        append_skill_usage(
            REPO_ROOT,
            task=task,
            iteration=iteration,
            loaded_skills=loaded_skills(REPO_ROOT),
            used_skills=skills_used,
            triggered_by=strategy,
            result="new_best" if improved else "no_best_improvement",
            score_delta_after_use=score_improvement,
        )
        last_record = record
        print(
            f"Task {task} iter {iteration}: {strategy} "
            f"valid={eval_result.valid} score={eval_result.score:.6f} "
            f"sum={float(record['sum_radii'] or 0.0):.6f} improved={improved}"
        )

    exported = archive.export_best_code(task, adapter.solution_path(task))
    if exported is None:
        fallback = generator.generate(task, "baseline_safe_grid", iteration=999)
        adapter.write_candidate(task, fallback.code)
        exported = adapter.solution_path(task)
    final_eval = adapter.evaluate_task(task)
    append_final_summary(
        human_log,
        f"exported_solution: {exported}\n"
        f"final_valid: {final_eval.valid}\n"
        f"final_score: {final_eval.score:.6f}\n"
        f"final_sum_radii: {float(final_eval.sum_radii or 0.0):.6f}\n"
        f"failure_type: {final_eval.failure_type}\n"
        f"skills_used: static-export, evaluator-feedback, archive-observability\n",
    )


def skills_for_iteration(strategy: str, eval_result=None, effective_failure: Optional[str] = None) -> list:
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
    if effective_failure in {"overlap", "boundary_violation", "negative_radius", "nonfinite", "perimeter_error"}:
        if "packing-repair" not in skills:
            skills.append("packing-repair")
    return skills


def choose_strategy(task: str, iteration: int, best_record: Optional[Dict],
                    last_record: Optional[Dict], no_improve: int,
                    use_benchmark_seeds: bool = False,
                    benchmark_attempted: bool = False,
                    refine_benchmark: bool = False,
                    strategy_counts: Optional[Dict[str, int]] = None) -> str:
    strategy_counts = strategy_counts or {}
    best_score = float(best_record.get("score") or 0.0) if best_record else 0.0
    plateau = no_improve >= 2
    if (not benchmark_attempted) and (use_benchmark_seeds or best_score < 0.999 or plateau):
        return BENCHMARK_SEED_STRATEGY
    if refine_benchmark and best_record is not None:
        if int(strategy_counts.get("fixed_centers_radius_lp") or 0) == 0:
            return "fixed_centers_radius_lp"
        if int(strategy_counts.get("micro_perturb_lp_refine") or 0) == 0:
            return "micro_perturb_lp_refine"
        if task == "A" and int(strategy_counts.get("optional_fico_task_a_seed") or 0) == 0:
            return "optional_fico_task_a_seed"
        return "micro_perturb_lp_refine"
    if iteration == 0:
        return "baseline_safe_grid"
    if best_record is None:
        if iteration == 1:
            return "hexagonal_or_staggered_initialization"
        return "scipy_slsqp_joint"
    if last_record and not last_record.get("valid"):
        failure = last_record.get("failure_type")
        if failure in {"overlap", "boundary_violation", "nonfinite", "shape_error", "perimeter_error", "negative_radius"}:
            return "hexagonal_or_staggered_initialization"
    if no_improve >= 2:
        return "perturb_best_and_repair"
    cycle = [
        "hexagonal_or_staggered_initialization",
        "scipy_slsqp_joint",
        "multi_start_slsqp",
        "perturb_best_and_repair",
    ]
    return cycle[(iteration - 1) % len(cycle)]


def decision_after_result(valid: bool, failure_type: str, score: float,
                          previous_best: Optional[Dict]) -> str:
    if not valid:
        if failure_type == "overlap":
            return "Increase safety margin and prefer repair/staggered generation next."
        if failure_type == "boundary_violation":
            return "Tighten boundary constraints and use structured initialization next."
        return "Fall back to conservative generation or shorter SLSQP restart."
    best_score = float(previous_best.get("score") or 0.0) if previous_best else 0.0
    if score > best_score:
        return "Archive improved; next step can exploit this candidate with perturbation or diversify with multi-start."
    if failure_type == "low_score":
        return "Valid but below the current target band; use multi-start or perturbation to seek higher sum radii."
    if failure_type == "plateau":
        return "Valid plateau detected; perturb the best candidate or diversify structured starts."
    return "Valid but no improvement; diversify initialization or perturb current best."


def build_observation(task: str, iteration: int, best_record: Optional[Dict],
                      last_record: Optional[Dict], no_improve: int,
                      archive: ArchiveManager) -> Dict:
    return {
        "task": task,
        "iteration": iteration,
        "last_evaluator_result": compact_record(last_record),
        "best_archive_state": compact_record(best_record),
        "no_improve_count": no_improve,
        "strategy_stats": archive.strategy_stats(task),
    }


def compact_record(record: Optional[Dict]) -> Optional[Dict]:
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
    return {k: record.get(k) for k in keys if k in record}


def strategy_reason(strategy: str, observation: Dict, llm_reason: str) -> str:
    if llm_reason and "disabled" not in llm_reason.lower():
        return llm_reason
    last = observation.get("last_evaluator_result") or {}
    if strategy == "baseline_safe_grid":
        return "Start from a conservative valid fallback to seed archive memory."
    if strategy == "hexagonal_or_staggered_initialization":
        return "Use a structured layout after failure, low score, or diversification need."
    if strategy == "scipy_slsqp_joint":
        return "Optimize centers and radii jointly with evaluator-governed validation."
    if strategy == "multi_start_slsqp":
        return "Run multiple deterministic starts because current archive can be improved."
    if strategy == "perturb_best_and_repair":
        return "Exploit the best archive candidate after plateau or no-improvement rounds."
    if strategy == BENCHMARK_SEED_STRATEGY:
        return "Use public DominikKamp/Packing geometry as an external benchmark warm-start candidate, then rely on the official evaluator."
    if strategy == "fixed_centers_radius_lp":
        return "Hold the current best centers fixed and solve the radius LP, accepting only official-evaluator-valid improvements."
    if strategy == "micro_perturb_lp_refine":
        return "Try tiny perturbations around the current best geometry, solve a radius LP, and rely on official validation."
    if strategy == "optional_fico_task_a_seed":
        return "Try the optional public FICO Problem 13 Task A seed if a local copy is available, then rely on official validation."
    return f"Selected by policy from last failure {last.get('failure_type', 'none')}."


def action_summary(strategy: str) -> str:
    mapping = {
        "baseline_safe_grid": "fixed-center LP radius solve with safe grid fallback",
        "hexagonal_or_staggered_initialization": "structured center generation plus LP radius repair",
        "scipy_slsqp_joint": "SLSQP joint optimization followed by LP safety repair",
        "multi_start_slsqp": "deterministic multi-start SLSQP and best-valid export",
        "perturb_best_and_repair": "archive-best perturbation with LP radius recomputation",
        BENCHMARK_SEED_STRATEGY: "public benchmark geometry conversion to static candidate plus official validation",
        "fixed_centers_radius_lp": "fixed-center LP radius maximization near the benchmark candidate",
        "micro_perturb_lp_refine": "tiny center/width perturbation followed by fixed-center LP",
        "optional_fico_task_a_seed": "optional public FICO Problem 13 seed conversion plus official validation",
    }
    return mapping.get(strategy, "candidate generation and official evaluator validation")


def effective_failure_type(eval_result, score_improvement: float, no_improve: int) -> str:
    if not eval_result.valid:
        return eval_result.failure_type or "unknown"
    if no_improve >= 2 and score_improvement <= 1e-9:
        return "plateau"
    if eval_result.score < 0.90:
        return "low_score"
    return "none"


def reset_skill_usage(repo_root: Path, run_id: str) -> None:
    path = repo_root / "agent" / "skills" / "usage_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "loaded_skills": loaded_skills(repo_root),
                "iterations": [],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def loaded_skills(repo_root: Path) -> list:
    skills_dir = repo_root / "agent" / "skills"
    if not skills_dir.exists():
        return []
    return sorted(path.parent.name for path in skills_dir.glob("*/SKILL.md"))


def append_skill_usage(repo_root: Path, task: str, iteration: int, loaded_skills: list,
                       used_skills: list, triggered_by: str, result: str,
                       score_delta_after_use: float) -> None:
    path = repo_root / "agent" / "skills" / "usage_stats.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"loaded_skills": loaded_skills, "iterations": []}
    payload["loaded_skills"] = loaded_skills
    payload.setdefault("iterations", []).append(
        {
            "task": task,
            "iteration": int(iteration),
            "loaded_skills": loaded_skills,
            "used_skills": used_skills,
            "triggered_by": triggered_by,
            "result": result,
            "score_delta_after_use": float(score_delta_after_use),
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_strategy_portfolio_metrics(repo_root: Path, stats: Dict, lineage_paths: Dict[str, str]) -> Path:
    path = repo_root / "agent" / "archive" / "metrics" / "strategy_portfolio.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_portfolio_stats": stats,
        "lineage_paths": lineage_paths,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_human_agent_division(repo_root: Path, archive: ArchiveManager) -> None:
    summary = {
        "human_provided": [
            {
                "item": "Problem understanding and scoring constraints",
                "evidence_artifacts": ["task_A/task_description.md", "task_B/task_description.md"],
            },
            {
                "item": "Architecture preferences: no large frameworks, no role-play personas, preserve official evaluators",
                "evidence_artifacts": ["submission/report.md", "agent/run.py"],
            },
            {
                "item": "Permission to use public benchmark seed data",
                "evidence_artifacts": ["benchmarks/dominikkamp/SOURCE.md"],
            },
        ],
        "agent_completed": [
            {
                "item": "Candidate code generation and static export",
                "evidence_artifacts": ["task_A/solution.py", "task_B/solution.py"],
            },
            {
                "item": "Official evaluator invocation and raw output archival",
                "evidence_artifacts": ["submission/agent/run_archive.jsonl", "agent/archive/metrics/run_log.jsonl"],
            },
            {
                "item": "Benchmark seed conversion and metadata tracking",
                "evidence_artifacts": ["agent/benchmark_seeds.py", "benchmarks/dominikkamp/"],
            },
            {
                "item": "Best-valid export, lineage DAG, strategy portfolio, and safety audit",
                "evidence_artifacts": [
                    "agent/archive/lineage/task_A_best_lineage.json",
                    "agent/archive/lineage/task_B_best_lineage.json",
                    "agent/archive/metrics/strategy_portfolio.json",
                    "submission/safety_report.json",
                ],
            },
        ],
        "best_candidates": {
            task: (archive.best_record(task) or {}).get("candidate_id")
            for task in ("A", "B")
        },
    }
    json_path = repo_root / "submission" / "human_agent_division.json"
    md_path = repo_root / "submission" / "human_agent_division.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Human-Agent Division Audit", ""]
    lines.append("## Human Provided")
    for item in summary["human_provided"]:
        lines.append(f"- {item['item']} Evidence: {', '.join(item['evidence_artifacts'])}")
    lines.append("")
    lines.append("## Agent Completed")
    for item in summary["agent_completed"]:
        lines.append(f"- {item['item']} Evidence: {', '.join(item['evidence_artifacts'])}")
    lines.append("")
    lines.append("## Best Candidates")
    for task, candidate_id in summary["best_candidates"].items():
        lines.append(f"- Task {task}: `{candidate_id}`")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_demo_builder(repo_root: Path) -> None:
    script = repo_root / "scripts" / "build_demo_data.py"
    if not script.exists():
        return
    subprocess.run([sys.executable, str(script)], cwd=str(repo_root), check=False)


def create_submission(repo_root: Path, archive: ArchiveManager, evaluate_all_output: str,
                      safety_guard: SafetyGuard) -> None:
    submission = repo_root / "submission"
    submission.mkdir(exist_ok=True)

    agent_dst = submission / "agent"
    shutil.copytree(
        repo_root / "agent",
        agent_dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    skills_src = repo_root / "skills"
    if skills_src.exists():
        shutil.copytree(skills_src, submission / "skills", dirs_exist_ok=True)
    project_skills_src = repo_root / "agent" / "skills"
    if project_skills_src.exists():
        shutil.copytree(project_skills_src, submission / "agent" / "skills", dirs_exist_ok=True)

    requirements = repo_root / "requirements.txt"
    if requirements.exists():
        shutil.copyfile(requirements, submission / "requirements.txt")
    benchmarks_src = repo_root / "benchmarks"
    if benchmarks_src.exists():
        shutil.copytree(benchmarks_src, submission / "benchmarks", dirs_exist_ok=True)
    shutil.copyfile(archive.archive_jsonl, agent_dst / "run_archive.jsonl")
    summary = archive.root / "summary.json"
    if summary.exists():
        shutil.copyfile(summary, agent_dst / "run_summary.json")

    for task in ("A", "B"):
        src_dir = repo_root / ("task_A" if task == "A" else "task_B")
        dst_dir = submission / ("task_A" if task == "A" else "task_B")
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_dir / "solution.py", dst_dir / "solution.py")
        log_name = "run_log_a.log" if task == "A" else "run_log_b.log"
        if (src_dir / log_name).exists():
            shutil.copyfile(src_dir / log_name, dst_dir / log_name)

    write_human_agent_division(repo_root, archive)
    generate_report(repo_root, archive, submission / "report.md", evaluate_all_output)
    safety_guard.check_post_run(archive.run_id, write_submission=True)
    generate_report(repo_root, archive, submission / "report.md", evaluate_all_output)
    run_demo_builder(repo_root)
    safety_guard.check_post_run(archive.run_id, write_submission=True)


if __name__ == "__main__":
    raise SystemExit(main())
