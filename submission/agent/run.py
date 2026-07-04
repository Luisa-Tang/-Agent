"""Main entrypoint for the local circle-packing Agent."""

from __future__ import annotations

import argparse
import shutil
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
from candidate_generators import CandidateGenerator
from evaluator_adapter import EvaluatorAdapter
from geometry_utils import PackingData, safety_metrics
from llm_reflector import LLMReflector
from log_utils import append_final_summary, append_human_iteration, reset_human_log
from report_data import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Agent for AlgorithmOptimization circle packing")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Use fewer starts and lower optimizer iteration budgets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=None, help="Optional whole-run wall clock limit in seconds")
    parser.add_argument("--use-llm", action="store_true", help="Enable optional LLM strategy reflection")
    parser.add_argument("--llm-base-url", default="http://10.12.111.139/v1")
    parser.add_argument("--llm-model", default="glm-5.2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive = ArchiveManager(REPO_ROOT, run_id)
    adapter = EvaluatorAdapter(REPO_ROOT, python_executable=sys.executable)
    generator = CandidateGenerator(seed=args.seed, fast=args.fast)
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
        f"use_llm={args.use_llm}, llm_model={args.llm_model}"
    )

    start = time.time()
    for task in tasks:
        read_task_context(task)
        run_task_loop(task, args, generator, adapter, archive, reflector, start)

    summary_path = archive.write_summary()
    evaluate_all_proc = adapter.run_evaluate_all(filename="solution.py")
    evaluate_all_output = evaluate_all_proc.stdout
    if evaluate_all_proc.stderr:
        evaluate_all_output += "\n[stderr]\n" + evaluate_all_proc.stderr
    print(evaluate_all_output)
    print(f"archive_summary={summary_path}")

    create_submission(REPO_ROOT, archive, evaluate_all_output)
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
        local_strategy = choose_strategy(iteration, archive.best_record(task), last_record, no_improve)
        llm_decision = reflector.decide(
            task=task,
            iteration=iteration,
            best_record=previous_best,
            last_record=last_record,
            no_improve=no_improve,
            local_strategy=local_strategy,
            recent_records=archive.records_for_task(task),
        )
        strategy = llm_decision.strategy or local_strategy
        thought = {
            "selected_strategy": strategy,
            "local_policy_strategy": local_strategy,
            "reason": strategy_reason(strategy, observation, llm_decision.reason),
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
        action = {
            "strategy": strategy,
            "optimizer_or_repair": action_summary(strategy),
            "generated_code_bytes": len(candidate.code.encode("utf-8")),
            "geometry_metrics": metrics,
            "diagnostics": candidate.diagnostics,
        }
        candidate_id = archive.make_candidate_id(task, iteration, strategy)
        eval_result = adapter.evaluate_task(task, candidate.code)
        code_snapshot = archive.save_candidate_code(task, candidate_id, candidate.code)
        raw_output = archive.save_raw_output(task, candidate_id, eval_result.raw_output)

        score_improvement = float(eval_result.score) - float(previous_best.get("score") or 0.0) if previous_best else float(eval_result.score)
        effective_failure = effective_failure_type(eval_result, score_improvement, no_improve)
        next_observation = {
            "valid": eval_result.valid,
            "score": eval_result.score,
            "sum_radii": eval_result.sum_radii if eval_result.sum_radii is not None else candidate.data.sum_radii,
            "failure_type": effective_failure,
            "raw_output": raw_output,
            "returncode": eval_result.returncode,
            "elapsed": eval_result.elapsed_text,
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
            "decision": decision_after_result(eval_result.valid, effective_failure,
                                              eval_result.score, previous_best),
            "raw_output": raw_output,
            "code_snapshot": code_snapshot,
            "diagnostics": candidate.diagnostics,
            "geometry_metrics": metrics,
            "local_policy_strategy": local_strategy,
            "llm_decision": llm_decision.to_record(),
            "trace": {
                "observation": observation,
                "thought_decision": thought,
                "action": action,
                "next_observation": next_observation,
            },
        }
        archive.add_record(record)
        current_best = archive.best_record(task)
        improved = bool(current_best and current_best.get("candidate_id") == candidate_id)
        if improved:
            best_data = candidate.data
            no_improve = 0
        else:
            no_improve += 1
        append_human_iteration(human_log, record, improved=improved)
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
        f"failure_type: {final_eval.failure_type}\n",
    )


def choose_strategy(iteration: int, best_record: Optional[Dict],
                    last_record: Optional[Dict], no_improve: int) -> str:
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
    return f"Selected by policy from last failure {last.get('failure_type', 'none')}."


def action_summary(strategy: str) -> str:
    mapping = {
        "baseline_safe_grid": "fixed-center LP radius solve with safe grid fallback",
        "hexagonal_or_staggered_initialization": "structured center generation plus LP radius repair",
        "scipy_slsqp_joint": "SLSQP joint optimization followed by LP safety repair",
        "multi_start_slsqp": "deterministic multi-start SLSQP and best-valid export",
        "perturb_best_and_repair": "archive-best perturbation with LP radius recomputation",
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


def create_submission(repo_root: Path, archive: ArchiveManager, evaluate_all_output: str) -> None:
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

    requirements = repo_root / "requirements.txt"
    if requirements.exists():
        shutil.copyfile(requirements, submission / "requirements.txt")
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

    generate_report(repo_root, archive, submission / "report.md", evaluate_all_output)


if __name__ == "__main__":
    raise SystemExit(main())
