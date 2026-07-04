"""Optional LangGraph orchestration entrypoint for the GeoOpt Agent.

The stable `agent/run.py` pipeline remains the default. This runner adds an
explicit StateGraph around the same evaluator, generator, archive, strategy
controller, and safety guard modules.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from archive import ArchiveManager
from candidate_generators import CandidateGenerator
from evaluator_adapter import EvaluatorAdapter
from graph_nodes import (
    GraphRuntime,
    evaluate_candidate,
    generate_candidate,
    load_task,
    observe_archive,
    parse_feedback,
    safety_check,
    select_strategy,
    static_export,
    update_archive,
)
from graph_routes import route_next
from lineage import write_best_lineages
from run import create_submission, write_strategy_portfolio_metrics
from safety_guard import SafetyGuard
from state import GeoOptState
from strategy_controller import StrategyPortfolioController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional LangGraph runner for GeoOpt Agent")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--use-benchmark-seeds", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    langgraph, import_error = _load_langgraph()
    if langgraph is None:
        print(
            "LangGraph is not importable in this Python environment. "
            "Install a Python-compatible `langgraph` package to run agent/langgraph_runner.py. "
            "The stable agent/run.py fallback remains available.",
            file=sys.stderr,
        )
        if import_error:
            print(f"import_error={import_error!r}", file=sys.stderr)
        return 2

    run_id = datetime.utcnow().strftime("langgraph_%Y%m%d_%H%M%S")
    archive = ArchiveManager(REPO_ROOT, run_id)
    adapter = EvaluatorAdapter(REPO_ROOT, python_executable=sys.executable)
    generator = CandidateGenerator(seed=args.seed, fast=args.fast)
    controller = StrategyPortfolioController()
    safety_guard = SafetyGuard(REPO_ROOT)
    safety_guard.capture_pre_run()
    metrics_graph_run_log = REPO_ROOT / "agent" / "archive" / "metrics" / "langgraph_run_log.jsonl"
    metrics_graph_run_log.parent.mkdir(parents=True, exist_ok=True)
    metrics_graph_run_log.write_text("", encoding="utf-8")

    tasks = ["A", "B"] if args.task == "both" else [args.task]
    print(f"repo_root={REPO_ROOT}")
    print(f"run_id={run_id}")
    print(f"python={sys.executable}")
    print(
        f"tasks={tasks}, iterations={args.iterations}, seed={args.seed}, "
        f"fast={args.fast}, use_benchmark_seeds={args.use_benchmark_seeds}"
    )

    last_graph = None
    for task in tasks:
        runtime = GraphRuntime(
            repo_root=REPO_ROOT,
            run_id=run_id,
            seed=args.seed,
            max_iterations=max(0, int(args.iterations)),
            fast=bool(args.fast),
            use_benchmark_seeds=bool(args.use_benchmark_seeds),
            archive=archive,
            adapter=adapter,
            generator=generator,
            controller=controller,
            safety_guard=safety_guard,
            task=task,
            graph_run_log=archive.root / f"task_{task}_run_log.jsonl",
            metrics_graph_run_log=metrics_graph_run_log,
        )
        graph = build_graph(langgraph, runtime)
        last_graph = graph
        initial_state: GeoOptState = {
            "task": task,
            "iteration": 0,
            "max_iterations": max(0, int(args.iterations)),
            "seed": int(args.seed),
            "archive_summary": {},
            "strategy_stats": {},
            "selected_strategy": None,
            "decision_reason": "",
            "candidate_id": None,
            "parent_candidate_id": None,
            "eval_result": None,
            "failure_type": None,
            "best_candidate_id": None,
            "best_score": 0.0,
            "best_sum_radii": 0.0,
            "next_action": "load_task",
            "skills_used": [],
            "artifacts": {"benchmark_allowed": bool(args.use_benchmark_seeds)},
        }
        config = {"configurable": {"thread_id": f"geoopt-task{task}-seed{args.seed}"}}
        final_state = graph.invoke(initial_state, config=config)
        print(
            f"Task {task}: best={final_state.get('best_candidate_id')} "
            f"score={float(final_state.get('best_score') or 0.0):.6f} "
            f"sum={float(final_state.get('best_sum_radii') or 0.0):.6f}"
        )

    summary_path = archive.write_summary()
    lineage_paths = write_best_lineages(REPO_ROOT, archive.records)
    write_strategy_portfolio_metrics(REPO_ROOT, controller.stats(archive.records), lineage_paths)
    export_graph_visualization(REPO_ROOT, last_graph)
    evaluate_all_proc = adapter.run_evaluate_all(filename="solution.py")
    evaluate_all_output = evaluate_all_proc.stdout
    if evaluate_all_proc.stderr:
        evaluate_all_output += "\n[stderr]\n" + evaluate_all_proc.stderr
    print(evaluate_all_output)
    create_submission(REPO_ROOT, archive, evaluate_all_output, safety_guard)
    export_graph_visualization(REPO_ROOT, last_graph)
    _copy_graph_runner_artifacts(REPO_ROOT)
    print(f"archive_summary={summary_path}")
    print(f"submission_dir={REPO_ROOT / 'submission'}")
    return 0 if evaluate_all_proc.returncode == 0 else evaluate_all_proc.returncode


def _load_langgraph() -> Tuple[Any, Exception]:
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - only exercised when optional dependency is absent
        return None, exc
    return {"StateGraph": StateGraph, "START": START, "END": END, "MemorySaver": MemorySaver}, None


def build_graph(langgraph: Dict[str, Any], runtime: GraphRuntime):
    builder = langgraph["StateGraph"](GeoOptState)
    builder.add_node("load_task", lambda state: load_task(state, runtime))
    builder.add_node("observe_archive", lambda state: observe_archive(state, runtime))
    builder.add_node("select_strategy", lambda state: select_strategy(state, runtime))
    builder.add_node("generate_candidate", lambda state: generate_candidate(state, runtime))
    builder.add_node("evaluate_candidate", lambda state: evaluate_candidate(state, runtime))
    builder.add_node("parse_feedback", lambda state: parse_feedback(state, runtime))
    builder.add_node("update_archive", lambda state: update_archive(state, runtime))
    builder.add_node("static_export", lambda state: static_export(state, runtime))
    builder.add_node("safety_check", lambda state: safety_check(state, runtime))

    builder.add_edge(langgraph["START"], "load_task")
    builder.add_edge("load_task", "observe_archive")
    builder.add_edge("observe_archive", "select_strategy")
    builder.add_edge("select_strategy", "generate_candidate")
    builder.add_edge("generate_candidate", "evaluate_candidate")
    builder.add_edge("evaluate_candidate", "parse_feedback")
    builder.add_edge("parse_feedback", "update_archive")
    builder.add_conditional_edges(
        "update_archive",
        route_next,
        {
            "select_strategy": "select_strategy",
            "generate_candidate": "generate_candidate",
            "static_export": "static_export",
            "safety_check": "safety_check",
            "end": langgraph["END"],
        },
    )
    builder.add_edge("static_export", "safety_check")
    builder.add_edge("safety_check", langgraph["END"])

    try:
        return builder.compile(checkpointer=langgraph["MemorySaver"]())
    except TypeError:
        return builder.compile()


def export_graph_visualization(repo_root: Path, graph=None) -> None:
    demo_dir = repo_root / "submission" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    mermaid = None
    if graph is not None:
        try:
            mermaid = graph.get_graph().draw_mermaid()
        except Exception:
            mermaid = None
    if not mermaid:
        mermaid = """flowchart TD
    START([START]) --> load_task
    load_task --> observe_archive
    observe_archive --> select_strategy
    select_strategy --> generate_candidate
    generate_candidate --> evaluate_candidate
    evaluate_candidate --> parse_feedback
    parse_feedback --> update_archive
    update_archive -->|continue| select_strategy
    update_archive -->|repair / benchmark / explore / exploit| generate_candidate
    update_archive -->|iteration >= max_iterations| static_export
    static_export --> safety_check
    safety_check --> END([END])
"""
    text = """GeoOpt LangGraph StateGraph

START -> load_task -> observe_archive -> select_strategy -> generate_candidate
generate_candidate -> evaluate_candidate -> parse_feedback -> update_archive
update_archive routes conditionally:
- select_strategy: normal portfolio decision
- generate_candidate: repair, benchmark warm-start, plateau exploration, or best-candidate exploitation
- static_export: iteration budget reached
static_export -> safety_check -> END

Nodes reuse existing modules:
- evaluator_adapter.py
- candidate_generators.py
- archive.py
- strategy_controller.py
- safety_guard.py
"""
    (demo_dir / "agent_graph.mmd").write_text(mermaid, encoding="utf-8")
    (demo_dir / "agent_graph.txt").write_text(text, encoding="utf-8")


def _copy_graph_runner_artifacts(repo_root: Path) -> None:
    agent_dst = repo_root / "submission" / "agent"
    agent_dst.mkdir(parents=True, exist_ok=True)
    for name in ("graph_nodes.py", "graph_routes.py", "langgraph_runner.py", "state.py"):
        src = repo_root / "agent" / name
        if src.exists():
            shutil.copyfile(src, agent_dst / name)


if __name__ == "__main__":
    raise SystemExit(main())
