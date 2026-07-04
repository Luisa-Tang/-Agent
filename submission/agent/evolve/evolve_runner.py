"""Optional GeoEvolve-lite self-evolution harness."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from archive import ArchiveManager, sanitize_json
from contact_graph import summarize_contact_graph
from evaluator_adapter import EvaluatorAdapter
from geometry_utils import PackingData, TASK_SPECS
from novelty_archive import NoveltyArchive

try:
    from .cascade_evaluator import CascadeEvaluator
    from .mutation_operators import create_child_program, create_seed_program
    from .novelty_filter import NoveltyFilter
    from .program_db import ProgramDatabase, ProgramRecord
    from .prompt_sampler import PromptSampler
    from .strategy_bandit import StrategyBandit, V2_OPERATORS
except ImportError:  # pragma: no cover - direct module execution fallback
    from cascade_evaluator import CascadeEvaluator
    from mutation_operators import create_child_program, create_seed_program
    from novelty_filter import NoveltyFilter
    from program_db import ProgramDatabase, ProgramRecord
    from prompt_sampler import PromptSampler
    from strategy_bandit import StrategyBandit, V2_OPERATORS


@dataclass
class SelfEvolveConfig:
    generations: int = 20
    batch_size: int = 4
    max_official_evals: int = 40
    novelty_threshold: float = 0.25
    seed: int = 42
    use_benchmark_seeds: bool = True
    use_llm: bool = False
    evolve_blocks_v2: bool = False
    parent_sampling: str = "balanced"
    enable_block_crossover: bool = False
    geometry_novelty_threshold: float = 0.35


def run_self_evolution_search(repo_root: Path, tasks: List[str], archive: ArchiveManager,
                              adapter: EvaluatorAdapter,
                              config: SelfEvolveConfig) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    program_db = ProgramDatabase(repo_root, archive.run_id)
    operators = list(V2_OPERATORS) if config.evolve_blocks_v2 else None
    if operators and not config.enable_block_crossover:
        operators = [name for name in operators if name != "block_crossover"]
    bandit = StrategyBandit(operators=operators)
    prompt_sampler = PromptSampler(repo_root, program_db.prompt_root)
    cascade = CascadeEvaluator(repo_root, adapter, archive, program_db)
    cascade.reset_log()
    novelty = NoveltyFilter(
        threshold=float(config.geometry_novelty_threshold if config.evolve_blocks_v2 else config.novelty_threshold)
    )
    novelty_archive = NoveltyArchive(repo_root)
    novelty_archive.add_many(archive.records)
    rng = np.random.default_rng(int(config.seed))
    program_data: Dict[str, PackingData] = {}
    official_count = 0
    generated_count = 0
    accepted_count = 0
    task_summaries: Dict[str, Any] = {}

    for task in tasks:
        task = task.upper()
        current_data = _load_current_solution(repo_root, task)
        contact = summarize_contact_graph(task, current_data.centers, current_data.radii, current_data.width, current_data.height, tolerance=5e-8)
        best_before = archive.best_record(task)
        seed_metadata = {
            "score": float(best_before.get("score") or current_data.score) if best_before else current_data.score,
            "sum_radii": float(best_before.get("sum_radii") or current_data.sum_radii) if best_before else current_data.sum_radii,
            "contact_graph_hash": contact.get("contact_graph_hash"),
            "boundary_pattern": contact.get("active_boundary_pattern"),
            "source": "current best static solution",
            "best_candidate_id": best_before.get("candidate_id") if best_before else None,
        }
        seed_record = create_seed_program(program_db, task, seed_metadata)
        program_data[seed_record.program_id] = current_data
        novelty.remember(seed_record.code_hash, seed_record.contact_graph_hash, seed_record.boundary_pattern, seed_record.strategy_family, seed_record.block_hashes)
        task_summaries[task] = {
            "best_before": _compact_best(best_before, current_data),
            "seed_program_id": seed_record.program_id,
            "generated_programs": 1,
            "official_evals": 0,
            "valid_official": 0,
            "accepted_improvements": 0,
            "novelty_rejected": 0,
            "official_skipped": 0,
            "operator_attempts": {},
        }

    stop_reason = "generation_limit"
    max_generations = max(0, int(config.generations))
    batch_size = max(1, int(config.batch_size))
    max_official = max(0, int(config.max_official_evals))

    for generation in range(max_generations):
        if official_count >= max_official:
            stop_reason = "official_eval_budget"
            break
        for task in tasks:
            task = task.upper()
            for batch_index in range(batch_size):
                if official_count >= max_official:
                    stop_reason = "official_eval_budget"
                    break
                start = time.time()
                parent_record, parent_data, parent_mode = _select_parent(
                    program_db, program_data, task, generation, batch_index, rng, config
                )
                operator = bandit.select(generation * batch_size + batch_index, task)
                context = _operator_context(task, operator, generation, batch_index, config, rng, program_db, program_data)
                context["parent_sampling_mode"] = parent_mode
                context["parent_program_id"] = parent_record.program_id
                prompt = prompt_sampler.build_prompt(
                    task=task,
                    operator=operator,
                    parent_record=asdict(parent_record),
                    elites=[asdict(item) for item in program_db.diverse_elites(task, limit=2)],
                    recent_failures=[asdict(item) for item in _recent_failures(program_db, task, limit=2)],
                    operator_stats=bandit.to_dict(),
                    context=context,
                )
                next_id = program_db.next_program_id(task)
                prompt_path = prompt_sampler.save_prompt(next_id, prompt)
                child_record = create_child_program(program_db, task, parent_record, operator, rng, context, prompt_path=prompt_path)
                generated_count += 1
                task_summaries[task]["generated_programs"] += 1
                prepared = cascade.prepare(child_record, parent_data, context, novelty, config.novelty_threshold)
                novelty_score = float(prepared.novelty.novelty_score if prepared.novelty else 0.0)
                official_evaluated = False
                official_valid = False
                accepted = False
                delta = 0.0
                failure_type = prepared.failure_type

                if not prepared.eligible_for_official:
                    task_summaries[task]["novelty_rejected"] += 1 if prepared.failure_type == "rejected_novelty" else 0
                    cascade.log_official_skip(prepared, prepared.failure_type or "not_eligible_for_official", context)
                elif official_count >= max_official:
                    task_summaries[task]["official_skipped"] += 1
                    cascade.log_official_skip(prepared, "official_eval_budget_exhausted", context)
                else:
                    result = cascade.official_evaluate(prepared, archive.best_record(task))
                    official_evaluated = True
                    official_count += 1
                    task_summaries[task]["official_evals"] += 1
                    official_valid = bool(result.get("valid"))
                    failure_type = str(result.get("failure_type") or "none")
                    delta = float(result.get("score_improvement") or 0.0)
                    accepted = official_valid and delta > 0.0
                    if official_valid:
                        task_summaries[task]["valid_official"] += 1
                        program_data[child_record.program_id] = prepared.data
                        novelty_archive.add(_program_archive_record(child_record, prepared, result))
                    if accepted:
                        accepted_count += 1
                        task_summaries[task]["accepted_improvements"] += 1
                if prepared.data is not None:
                    novelty.remember(
                        child_record.code_hash,
                        (prepared.contact_graph or {}).get("contact_graph_hash"),
                        (prepared.contact_graph or {}).get("active_boundary_pattern"),
                        child_record.strategy_family,
                        child_record.block_hashes,
                    )
                bandit.record_attempt(
                    operator=operator,
                    accepted=accepted,
                    official_evaluated=official_evaluated,
                    valid=official_valid,
                    delta=delta,
                    runtime=time.time() - start,
                    novelty_score=novelty_score,
                    failure_type=failure_type,
                )
                counts = task_summaries[task]["operator_attempts"].setdefault(operator, 0)
                task_summaries[task]["operator_attempts"][operator] = counts + 1
            if official_count >= max_official:
                break

    for task in tasks:
        task = task.upper()
        archive.export_best_code(task, adapter.solution_path(task))
        best_after = archive.best_record(task)
        task_summaries[task]["best_after"] = _compact_best(best_after, _load_current_solution(repo_root, task))
        before_score = float((task_summaries[task].get("best_before") or {}).get("score") or 0.0)
        after_score = float((task_summaries[task].get("best_after") or {}).get("score") or 0.0)
        task_summaries[task]["improved_over_start"] = after_score > before_score + 1e-12
        task_summaries[task]["exceeded_denominator"] = after_score > 1.0
        task_summaries[task]["gap_to_denominator"] = max(0.0, 1.0 - after_score)

    novelty_path = novelty_archive.write()
    operator_stats_path = bandit.write(repo_root)
    tree_path = program_db.write_tree()
    block_metric_paths = program_db.write_block_metrics()
    summary = {
        "run_id": archive.run_id,
        "config": asdict(config),
        "stop_reason": stop_reason,
        "generated_program_count": generated_count + len(tasks),
        "novelty_rejected_count": int(novelty.rejected_count),
        "official_eval_count": official_count,
        "accepted_improvement_count": accepted_count,
        "tasks": task_summaries,
        "operator_stats": bandit.to_dict(),
        "program_db_path": str(program_db.path),
        "program_tree_path": str(tree_path),
        "operator_stats_path": str(operator_stats_path),
        "block_metrics_path": str(block_metric_paths["json"]),
        "block_metrics_jsonl_path": str(block_metric_paths["jsonl"]),
        "novelty_archive_path": str(novelty_path),
        "evolve_log_path": str(cascade.log_path),
    }
    summary_path = repo_root / "agent" / "archive" / "evolve" / "self_evolution_summary.json"
    summary_path.write_text(json.dumps(sanitize_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    write_self_evolution_report(repo_root, summary)
    if config.evolve_blocks_v2:
        write_evolve_blocks_v2_report(repo_root, summary)
    return summary


def _operator_context(task: str, operator: str, generation: int, batch_index: int,
                      config: SelfEvolveConfig, rng: np.random.Generator,
                      program_db: ProgramDatabase,
                      program_data: Dict[str, PackingData]) -> Dict[str, Any]:
    sigmas = [1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8]
    margins = [0.0, 1e-11, 3e-11, 1e-10, 3e-10]
    target_deltas = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]
    sigma = sigmas[(generation + batch_index) % len(sigmas)]
    context: Dict[str, Any] = {
        "task": task,
        "operator": operator,
        "generation": int(generation),
        "batch_index": int(batch_index),
        "seed": int(config.seed + 1009 * generation + 37 * batch_index + (1 if task == "A" else 2)),
        "sigma": float(sigma),
        "width_sigma": float(sigma),
        "lp_margin": float(margins[(generation + batch_index) % len(margins)]),
        "safety": 2e-10,
        "contact_tolerance": 5e-8,
        "contact_threshold": float([1e-7, 5e-8, 1e-8, 5e-9][(generation + batch_index) % 4]),
        "contact_scale": float([0.2, 0.35, 0.5][(generation + batch_index) % 3]),
        "solver": ["lp", "lp_safe", "lp_zero_margin"][(generation + batch_index) % 3],
        "solver_switch_jitter": float(sigma * 0.5),
        "target_delta": float(target_deltas[(generation + batch_index) % len(target_deltas)]),
        "width_direction": float(-1.0 if (generation + batch_index) % 2 else 1.0),
        "use_llm": bool(config.use_llm),
        "evolve_blocks_v2": bool(config.evolve_blocks_v2),
        "enable_fast_refine": bool(config.evolve_blocks_v2),
        "max_refine_steps": 6,
        "safety_shrink": 1e-10,
        "small_circle_count": 2 + ((generation + batch_index) % 2),
        "gap_probe_scale": float([0.02, 0.035, 0.05][(generation + batch_index) % 3]),
        "edge_band": float([2e-8, 1e-7, 5e-7][(generation + batch_index) % 3]),
        "aspect_delta": float(target_deltas[(generation + batch_index) % len(target_deltas)]),
    }
    if operator in {"crossover", "block_crossover"}:
        mate = _select_mate(program_db, program_data, task, generation, batch_index)
        if mate is not None:
            context["mate"] = _packing_payload(mate)
        context["mate_program_paths"] = _select_mate_program_paths(program_db, program_data, task, generation, batch_index)
    return context


def _select_parent(program_db: ProgramDatabase, program_data: Dict[str, PackingData],
                   task: str, generation: int, batch_index: int,
                   rng: np.random.Generator,
                   config: SelfEvolveConfig) -> Tuple[ProgramRecord, PackingData, str]:
    valid = [record for record in program_db.by_task(task) if record.valid and record.program_id in program_data]
    if not valid:
        raise RuntimeError(f"No valid parent program for Task {task}")
    mode = _parent_sampling_mode(config, rng)
    if mode == "exploit":
        record = max(valid, key=lambda item: (float(item.score), float(item.novelty_score)))
    elif mode == "diverse":
        elites = [item for item in program_db.diverse_elites(task, limit=8) if item.program_id in program_data]
        record = elites[(generation + batch_index) % len(elites)] if elites else valid[(generation + batch_index) % len(valid)]
    else:
        risky = sorted(valid, key=lambda item: (float(item.novelty_score), -float(item.score)), reverse=True)
        top = risky[: max(1, min(6, len(risky)))]
        record = top[int(rng.integers(0, len(top)))]
    return record, program_data[record.program_id], mode


def _parent_sampling_mode(config: SelfEvolveConfig, rng: np.random.Generator) -> str:
    if not config.evolve_blocks_v2:
        return "exploit" if rng.random() < 0.5 else "diverse"
    if config.parent_sampling == "exploit":
        return "exploit"
    if config.parent_sampling == "diverse":
        return "diverse"
    value = float(rng.random())
    if value < 0.50:
        return "exploit"
    if value < 0.80:
        return "diverse"
    return "risky_novelty"


def _select_mate(program_db: ProgramDatabase, program_data: Dict[str, PackingData],
                 task: str, generation: int, batch_index: int) -> Optional[PackingData]:
    candidates = [
        record for record in program_db.diverse_elites(task, limit=8)
        if record.valid and record.program_id in program_data
    ]
    if not candidates:
        return None
    return program_data[candidates[(generation + batch_index + 1) % len(candidates)].program_id]


def _select_mate_program_paths(program_db: ProgramDatabase, program_data: Dict[str, PackingData],
                               task: str, generation: int, batch_index: int) -> Dict[str, str]:
    candidates = [
        record for record in program_db.diverse_elites(task, limit=12)
        if record.valid and record.program_id in program_data
    ]
    if not candidates:
        return {}
    return {
        "geometry": str(candidates[(generation + batch_index) % len(candidates)].code_path),
        "radius": str(candidates[(generation + batch_index + 1) % len(candidates)].code_path),
        "refine": str(candidates[(generation + batch_index + 2) % len(candidates)].code_path),
    }


def _recent_failures(program_db: ProgramDatabase, task: str, limit: int) -> List[ProgramRecord]:
    failures = [record for record in program_db.by_task(task) if not record.valid]
    return failures[-limit:]


def _load_current_solution(repo_root: Path, task: str) -> PackingData:
    path = Path(repo_root) / ("task_A" if task.upper() == "A" else "task_B") / "solution.py"
    data = _packing_from_code_snapshot(path, task.upper())
    if data is None:
        raise RuntimeError(f"Could not load current Task {task} solution from {path}")
    return data


def _packing_from_code_snapshot(path: Path, task: str) -> Optional[PackingData]:
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_selfevolve_{task}_{abs(hash(str(path)))}", str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        n = int(TASK_SPECS[task]["n"])
        result = module.run_packing(n)
        if task == "A":
            centers, radii, width, height = result
            return PackingData(task=task, centers=np.asarray(centers, dtype=float), radii=np.asarray(radii, dtype=float), width=float(width), height=float(height))
        centers, radii, _sum = result
        return PackingData(task=task, centers=np.asarray(centers, dtype=float), radii=np.asarray(radii, dtype=float))
    except Exception:
        return None


def _packing_payload(data: PackingData) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "centers": np.asarray(data.centers, dtype=float),
        "radii": np.asarray(data.radii, dtype=float),
        "sum_radii": data.sum_radii,
        "score": data.score,
    }
    if data.task == "A":
        payload["width"] = float(data.width)
        payload["height"] = float(data.height)
    return payload


def _compact_best(record: Optional[Dict[str, Any]], fallback_data: Optional[PackingData]) -> Dict[str, Any]:
    if record:
        return {
            "candidate_id": record.get("candidate_id"),
            "strategy": record.get("strategy"),
            "score": record.get("score"),
            "sum_radii": record.get("sum_radii"),
            "width": record.get("width"),
            "height": record.get("height"),
            "contact_graph_hash": (record.get("contact_graph") or {}).get("contact_graph_hash"),
            "active_boundary_pattern": (record.get("contact_graph") or {}).get("active_boundary_pattern"),
        }
    data = fallback_data
    if data is None:
        return {}
    return {
        "candidate_id": "current_solution",
        "strategy": "current_static_solution",
        "score": data.score,
        "sum_radii": data.sum_radii,
        "width": data.width,
        "height": data.height,
    }


def _program_archive_record(record: ProgramRecord, prepared, official_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task": record.task,
        "candidate_id": official_result.get("candidate_id"),
        "parent_candidate_id": None,
        "strategy": f"self_evolve_{record.operator}",
        "strategy_family": record.strategy_family,
        "score": official_result.get("score"),
        "sum_radii": official_result.get("sum_radii"),
        "width": prepared.data.width if record.task == "A" else None,
        "height": prepared.data.height if record.task == "A" else None,
        "valid": official_result.get("valid"),
        "failure_type": official_result.get("failure_type"),
        "code_snapshot": official_result.get("code_snapshot"),
        "contact_graph": prepared.contact_graph,
        "geometry_metrics": prepared.geometry_metrics,
    }


def write_self_evolution_report(repo_root: Path, summary: Dict[str, Any]) -> Path:
    path = Path(repo_root) / "submission" / "self_evolution_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Self-Evolution Harness Report",
        "",
        "GeoEvolve-lite is an optional program-evolution harness. It is inspired by OpenEvolve, ShinkaEvolve, and CodeEvolve concepts, but it does not import those frameworks or replace the stable `agent/run.py` pipeline.",
        "",
        "## What Evolves",
        "",
        "The harness evolves candidate-generating programs and refinement operators. It does not evolve the final submitted coordinate arrays directly. A child program proposes centers, radii, and optional Task A width/height; the harness repairs and validates the geometry, then emits a static `solution.py` candidate only for official evaluation.",
        "",
        "## Program Database",
        "",
        f"- Program DB: `{_rel(repo_root, Path(summary.get('program_db_path', '')) )}`",
        f"- Program tree: `{_rel(repo_root, Path(summary.get('program_tree_path', '')) )}`",
        f"- Evolve log: `{_rel(repo_root, Path(summary.get('evolve_log_path', '')) )}`",
        f"- Operator stats: `{_rel(repo_root, Path(summary.get('operator_stats_path', '')) )}`",
        f"- Block metrics: `{_rel(repo_root, Path(summary.get('block_metrics_path', '')) )}`",
        "",
        "Each program record stores `program_id`, parent, task, operator, code path/hash, score, sum radii, official evaluator path, contact graph hash, boundary pattern, novelty score, strategy family, timestamp, and metadata.",
        "",
        "## Novelty Rejection and Cascade Evaluation",
        "",
        "- E0 checks syntax/import and the required `propose_candidate` function.",
        "- E1 executes the generator and runs internal geometry checks.",
        "- E2 computes quick score, contact graph, boundary pattern, code novelty, RMSD, and strategy-family novelty.",
        "- E3 runs official `evaluate.py` only for candidates that pass E0-E2 and the official-evaluation budget.",
        "",
        "## Strategy Bandit",
        "",
        "Operator selection uses a lightweight UCB-style controller over parameter mutation, solver switch, contact-threshold mutation, program patch fallback, crossover, and depth refinement. The score combines historical improvement, novelty, official validity, runtime penalty, repeated-failure penalty, and exploration bonus.",
        "",
        "## Results",
        "",
        f"- Stop reason: `{summary.get('stop_reason')}`",
        f"- Generated programs: `{summary.get('generated_program_count')}`",
        f"- Novelty rejected: `{summary.get('novelty_rejected_count')}`",
        f"- Official evaluate calls: `{summary.get('official_eval_count')}`",
        f"- Accepted improvements: `{summary.get('accepted_improvement_count')}`",
        "",
        "| Task | Best before | Best after | Improved | Exceeded 1.0 | Gap to 1.0 | Official evals | Valid official |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in sorted((summary.get("tasks") or {}).items()):
        before = item.get("best_before") or {}
        after = item.get("best_after") or {}
        lines.append(
            f"| {task} | {float(before.get('score') or 0.0):.12f} | "
            f"{float(after.get('score') or 0.0):.12f} | "
            f"{item.get('improved_over_start')} | {item.get('exceeded_denominator')} | "
            f"{float(item.get('gap_to_denominator') or 0.0):.3e} | "
            f"{int(item.get('official_evals') or 0)} | {int(item.get('valid_official') or 0)} |"
        )
    lines.extend(["", "## Operator Statistics", ""])
    lines.append("| Operator | Attempts | Accepted | Official evals | Valid | Best delta | Mean delta | Avg runtime | Novelty mean | Common failures |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for operator, stat in sorted((summary.get("operator_stats") or {}).items()):
        lines.append(
            f"| `{operator}` | {int(stat.get('attempts') or 0)} | {int(stat.get('accepted') or 0)} | "
            f"{int(stat.get('official_evaluated') or 0)} | {int(stat.get('valid_count') or 0)} | "
            f"{float(stat.get('best_delta') or 0.0):.3e} | {float(stat.get('mean_delta') or 0.0):.3e} | "
            f"{float(stat.get('avg_runtime') or 0.0):.3f} | {float(stat.get('novelty_mean') or 0.0):.3f} | "
            f"`{stat.get('common_failure_types') or {}}` |"
        )
    lines.extend(
        [
            "",
            "## Why This Still Matters If No Breakthrough Occurs",
            "",
            "A no-improvement run is still useful evidence: it records which program-level operators were tried, which candidates were rejected before expensive official evaluation, how many official evaluations were spent, and why the current static best was preserved. This is stronger evidence than repeatedly jittering coordinates around one contact graph.",
            "",
            "## Safety",
            "",
            "- Official evaluators and task descriptions are not modified.",
            "- Final `solution.py` files are overwritten only from the best official-valid archive candidate.",
            "- The submitted solutions remain static NumPy code with no network, LLM, or external-file dependency.",
            "- API keys are neither read from prompts nor written to archive, report, or solution artifacts.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_evolve_blocks_v2_report(repo_root: Path, summary: Dict[str, Any]) -> Path:
    repo_root = Path(repo_root)
    path = repo_root / "submission" / "evolve_blocks_v2_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    block_metrics = _load_json(Path(summary.get("block_metrics_path") or ""))
    block_stats = block_metrics.get("block_stats") or {}
    operator_stats = summary.get("operator_stats") or {}
    lines = [
        "# Evolve Blocks v2 Report",
        "",
        "Evolve Blocks v1 produced official-valid candidates but no accepted improvements. The v2 upgrade narrows evolution to named blocks: Task A aspect, geometry, radius, refine, and safety; Task B geometry, radius, refine, and safety.",
        "",
        "## What Changed",
        "",
        "- Candidate programs now expose named EVOLVE-BLOCK sections.",
        "- Mutation operators target geometry or refine blocks instead of rewriting the whole function.",
        "- Geometry novelty combines code-block novelty, contact graph change, boundary pattern change, center RMSD, radius-distribution change, and strategy-family novelty.",
        "- Parent sampling supports balanced exploit/diverse/risky-novelty modes.",
        "- `block_crossover` combines program blocks rather than splicing final coordinates.",
        "",
        "## Run Summary",
        "",
        f"- Generated programs: `{summary.get('generated_program_count')}`",
        f"- Novelty rejected: `{summary.get('novelty_rejected_count')}`",
        f"- Official evaluate calls: `{summary.get('official_eval_count')}`",
        f"- Accepted improvements: `{summary.get('accepted_improvement_count')}`",
        f"- Block metrics JSON: `{_rel(repo_root, Path(summary.get('block_metrics_path', '')) )}`",
        f"- Block metrics JSONL: `{_rel(repo_root, Path(summary.get('block_metrics_jsonl_path', '')) )}`",
        "",
        "| Task | Best before | Best after | Improved | Exceeded denominator | Gap to 1.0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, item in sorted((summary.get("tasks") or {}).items()):
        before = item.get("best_before") or {}
        after = item.get("best_after") or {}
        lines.append(
            f"| {task} | {float(before.get('score') or 0.0):.12f} | "
            f"{float(after.get('score') or 0.0):.12f} | "
            f"{item.get('improved_over_start')} | {item.get('exceeded_denominator')} | "
            f"{float(item.get('gap_to_denominator') or 0.0):.3e} |"
        )
    lines.extend(["", "## Block-Level Metrics", ""])
    lines.append("| Block changed | Attempts | Valid rate | Best delta | Accepted improvements | Mean contact changed | Mean boundary changed | Mean center RMSD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    if block_stats:
        for block, stat in sorted(block_stats.items()):
            lines.append(
                f"| `{block}` | {int(stat.get('attempts') or 0)} | "
                f"{float(stat.get('valid_rate') or 0.0):.3f} | "
                f"{float(stat.get('best_delta') or 0.0):.3e} | "
                f"{int(stat.get('accepted_improvements') or 0)} | "
                f"{float(stat.get('mean_contact_graph_changed') or 0.0):.3f} | "
                f"{float(stat.get('mean_boundary_pattern_changed') or 0.0):.3f} | "
                f"{float(stat.get('mean_centers_rmsd') or 0.0):.3e} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
    lines.extend(["", "## Operator Ablation", ""])
    lines.append("| Operator | Attempts | Official evals | Valid | Best delta | Mean delta | Novelty mean | Common failures |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for operator, stat in sorted(operator_stats.items()):
        lines.append(
            f"| `{operator}` | {int(stat.get('attempts') or 0)} | "
            f"{int(stat.get('official_evaluated') or 0)} | {int(stat.get('valid_count') or 0)} | "
            f"{float(stat.get('best_delta') or 0.0):.3e} | {float(stat.get('mean_delta') or 0.0):.3e} | "
            f"{float(stat.get('novelty_mean') or 0.0):.3f} | `{stat.get('common_failure_types') or {}}` |"
        )
    best_operator_line = _best_operator_interpretation(operator_stats)
    best_block_line = _best_block_interpretation(block_stats)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {best_operator_line}",
            f"- {best_block_line}",
            "- Final replacement policy did not change: no candidate can overwrite `solution.py` unless official evaluator score improves.",
            "- Official evaluator files and task descriptions are not modified.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _best_operator_interpretation(operator_stats: Dict[str, Any]) -> str:
    if not operator_stats:
        return "No operator statistics were recorded."
    best_by_delta = max(
        operator_stats.items(),
        key=lambda pair: (float(pair[1].get("best_delta") or 0.0), float(pair[1].get("novelty_mean") or 0.0)),
    )
    best_delta = float(best_by_delta[1].get("best_delta") or 0.0)
    if best_delta > 0.0:
        return f"Most effective operator by positive best delta: `{best_by_delta[0]}` ({best_delta:.3e})."
    closest = max(
        operator_stats.items(),
        key=lambda pair: float(pair[1].get("mean_delta") or 0.0),
    )
    return (
        "No operator produced a positive delta; closest average result was "
        f"`{closest[0]}` with mean delta {float(closest[1].get('mean_delta') or 0.0):.3e}."
    )


def _best_block_interpretation(block_stats: Dict[str, Any]) -> str:
    if not block_stats:
        return "No block statistics were recorded."
    best_by_delta = max(
        block_stats.items(),
        key=lambda pair: (float(pair[1].get("best_delta") or 0.0), float(pair[1].get("mean_contact_graph_changed") or 0.0)),
    )
    best_delta = float(best_by_delta[1].get("best_delta") or 0.0)
    if best_delta > 0.0:
        return f"Most promising block by positive best delta: `{best_by_delta[0]}` ({best_delta:.3e})."
    most_novel = max(
        block_stats.items(),
        key=lambda pair: (
            float(pair[1].get("mean_contact_graph_changed") or 0.0),
            float(pair[1].get("mean_boundary_pattern_changed") or 0.0),
        ),
    )
    return (
        "No block produced a positive delta; strongest geometry novelty came from "
        f"`{most_novel[0]}` with mean contact-change rate "
        f"{float(most_novel[1].get('mean_contact_graph_changed') or 0.0):.3f}."
    )


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(repo_root).resolve()))
    except Exception:
        return str(path)
