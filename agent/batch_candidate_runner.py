"""Batch breakthrough-search harness around the stable evaluator pipeline."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from archive import ArchiveManager, sanitize_json
from candidate_generators import GeneratedCandidate
from contact_graph import summarize_contact_graph
from contact_graph_refiner import contact_graph_feasibility_refine
from evaluator_adapter import EvaluatorAdapter
from geometry_utils import PackingData, TASK_SPECS, safety_metrics, validate_packing
from lineage import hash_packing_data, hash_text
from log_utils import utc_stamp
from novelty_archive import NoveltyArchive
from public_frontier_seeds import (
    frontier_source_summary,
    load_public_frontier_candidates,
    write_source_manifests,
)


@dataclass
class BreakthroughConfig:
    batch_size: int = 8
    max_candidates: int = 200
    seed: int = 42
    use_benchmark_seeds: bool = True
    target_score_a: float = 1.0
    target_score_b: float = 1.0


def run_breakthrough_search(repo_root: Path, tasks: List[str], archive: ArchiveManager,
                            adapter: EvaluatorAdapter, config: BreakthroughConfig) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    write_source_manifests(repo_root)
    log_path = repo_root / "agent" / "archive" / "metrics" / "breakthrough_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    novelty = NoveltyArchive(repo_root)
    novelty.add_many(archive.records)
    total_budget = max(0, int(config.max_candidates))
    per_task_budget = max(1, total_budget // max(1, len(tasks)))
    summary: Dict[str, Any] = {
        "run_id": archive.run_id,
        "config": config.__dict__,
        "frontier_sources": frontier_source_summary(repo_root),
        "tasks": {},
    }
    for task in tasks:
        task_summary = _run_task_breakthrough(
            repo_root=repo_root,
            task=task,
            archive=archive,
            adapter=adapter,
            novelty=novelty,
            log_path=log_path,
            config=config,
            max_candidates=per_task_budget,
        )
        summary["tasks"][task] = task_summary
        archive.export_best_code(task, adapter.solution_path(task))
    novelty_path = novelty.write()
    summary["novelty_archive_path"] = str(novelty_path)
    summary_path = repo_root / "agent" / "archive" / "metrics" / "breakthrough_summary.json"
    summary_path.write_text(json.dumps(sanitize_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_breakthrough_report(repo_root, summary, archive)
    return summary


def _run_task_breakthrough(repo_root: Path, task: str, archive: ArchiveManager,
                           adapter: EvaluatorAdapter, novelty: NoveltyArchive,
                           log_path: Path, config: BreakthroughConfig,
                           max_candidates: int) -> Dict[str, Any]:
    task = task.upper()
    generated_count = 0
    official_count = 0
    valid_count = 0
    improvements = 0
    best_before = archive.best_record(task)
    best_score_before = float(best_before.get("score") or 0.0) if best_before else 0.0
    target_score = float(config.target_score_a if task == "A" else config.target_score_b)
    parent_data = _load_current_solution(repo_root, task)
    parent_pool: List[PackingData] = [parent_data]
    frontier_verified = []
    source_failures = []

    if config.use_benchmark_seeds:
        for seed_candidate in load_public_frontier_candidates(repo_root, task):
            generated_count += 1
            result = _official_evaluate_candidate(
                repo_root, task, seed_candidate, archive, adapter,
                candidate_id=archive.make_candidate_id(task, 90000 + generated_count, seed_candidate.strategy),
                parent_record=archive.best_record(task),
                log_path=log_path,
                novelty=novelty,
                strategy_family="public_frontier_seed",
                archive_public_seed_only_if_valid=True,
            )
            official_count += 1
            if result["valid"]:
                valid_count += 1
                frontier_verified.append(result)
                parent_pool.append(seed_candidate.data)
                if result["improved"]:
                    improvements += 1
            else:
                source_failures.append(result)

    batch_size = max(1, int(config.batch_size))
    batch_index = 0
    while generated_count < max_candidates:
        parent = _select_parent(task, parent_pool, novelty, fallback=parent_data, batch_index=batch_index)
        batch = contact_graph_feasibility_refine(
            task,
            parent,
            iteration=batch_index,
            seed=int(config.seed) + batch_index,
            max_candidates=min(batch_size, max_candidates - generated_count),
        )
        generated_count += len(batch)
        internal_records = [_internal_record(task, candidate, batch_index, log_path) for candidate in batch]
        official_candidates = _select_top_internal(batch, internal_records, top_k=1)
        for candidate in official_candidates:
            official_count += 1
            result = _official_evaluate_candidate(
                repo_root, task, candidate, archive, adapter,
                candidate_id=archive.make_candidate_id(task, 91000 + official_count, candidate.strategy),
                parent_record=archive.best_record(task),
                log_path=log_path,
                novelty=novelty,
                strategy_family="contact_graph_refinement",
                archive_public_seed_only_if_valid=False,
            )
            if result["valid"]:
                valid_count += 1
                parent_pool.append(candidate.data)
                if result["improved"]:
                    improvements += 1
        batch_index += 1

    best_after = archive.best_record(task)
    best_score_after = float(best_after.get("score") or 0.0) if best_after else 0.0
    return {
        "task": task,
        "generated_count": generated_count,
        "official_evaluated_count": official_count,
        "valid_count": valid_count,
        "improvements": improvements,
        "best_before": _compact_best(best_before),
        "best_after": _compact_best(best_after),
        "best_score_before": best_score_before,
        "best_score_after": best_score_after,
        "target_score": target_score,
        "exceeded_target": best_score_after > target_score,
        "gap_to_target": max(0.0, target_score - best_score_after),
        "frontier_verified": frontier_verified,
        "frontier_failures": source_failures,
    }


def _internal_record(task: str, candidate: GeneratedCandidate, batch_index: int,
                     log_path: Path) -> Dict[str, Any]:
    valid, message = validate_packing(task, candidate.data.centers, candidate.data.radii, candidate.data.width, candidate.data.height, tol=1e-8)
    metrics = safety_metrics(candidate.data)
    contact = summarize_contact_graph(task, candidate.data.centers, candidate.data.radii, candidate.data.width, candidate.data.height, tolerance=5e-8)
    record = {
        "timestamp_utc": utc_stamp(),
        "phase": "internal_geometry_check",
        "task": task,
        "batch_index": batch_index,
        "strategy": candidate.strategy,
        "strategy_family": candidate.diagnostics.get("strategy_family") or candidate.strategy,
        "sum_radii": candidate.data.sum_radii,
        "score_estimate": candidate.data.score,
        "internal_valid": valid,
        "failure_type": "none" if valid else message,
        "geometry_metrics": metrics,
        "contact_graph": contact,
        "diagnostics": candidate.diagnostics,
    }
    _append_jsonl(log_path, record)
    return record


def _select_top_internal(candidates: List[GeneratedCandidate], records: List[Dict[str, Any]],
                         top_k: int) -> List[GeneratedCandidate]:
    ranked = sorted(
        zip(candidates, records),
        key=lambda pair: (
            bool(pair[1].get("internal_valid")),
            float(pair[1].get("sum_radii") or 0.0),
            float((pair[1].get("geometry_metrics") or {}).get("min_pairwise_margin") or -1.0),
        ),
        reverse=True,
    )
    return [candidate for candidate, _record in ranked[:max(1, top_k)]]


def _official_evaluate_candidate(repo_root: Path, task: str, candidate: GeneratedCandidate,
                                 archive: ArchiveManager, adapter: EvaluatorAdapter,
                                 candidate_id: str, parent_record: Optional[Dict[str, Any]],
                                 log_path: Path, novelty: NoveltyArchive,
                                 strategy_family: str,
                                 archive_public_seed_only_if_valid: bool) -> Dict[str, Any]:
    eval_result = adapter.evaluate_task(task, candidate.code)
    code_snapshot = archive.save_candidate_code(task, candidate_id, candidate.code)
    raw_output = archive.save_raw_output(task, candidate_id, eval_result.raw_output)
    previous_best = archive.best_record(task)
    previous_score = float(previous_best.get("score") or 0.0) if previous_best else 0.0
    score_improvement = float(eval_result.score) - previous_score
    contact = summarize_contact_graph(task, candidate.data.centers, candidate.data.radii, candidate.data.width, candidate.data.height, tolerance=5e-8)
    metrics = safety_metrics(candidate.data)
    source_metadata = candidate.diagnostics.get("source_metadata")
    if isinstance(source_metadata, dict):
        source_metadata = dict(source_metadata)
        source_metadata["official_evaluator_result"] = {
            "valid": eval_result.valid,
            "score": eval_result.score,
            "sum_radii": eval_result.sum_radii,
            "failure_type": eval_result.failure_type,
            "returncode": eval_result.returncode,
            "raw_output_path": str(raw_output),
            "exact_sum_radii": eval_result.exact_sum_radii,
            "exact_score": eval_result.exact_score,
        }
    record = {
        "task": task,
        "iteration": int(candidate_id.split("_")[1]) if "_" in candidate_id and candidate_id.split("_")[1].isdigit() else 0,
        "candidate_id": candidate_id,
        "strategy": candidate.strategy,
        "strategy_family": strategy_family,
        "parent_candidate_id": parent_record.get("candidate_id") if parent_record else None,
        "valid": eval_result.valid,
        "score": eval_result.score,
        "sum_radii": eval_result.sum_radii if eval_result.sum_radii is not None else candidate.data.sum_radii,
        "width": candidate.data.width if task == "A" else None,
        "height": candidate.data.height if task == "A" else None,
        "failure_type": eval_result.failure_type,
        "raw_failure_type": eval_result.failure_type,
        "score_improvement": score_improvement,
        "decision": "official-valid improvement" if eval_result.valid and score_improvement > 0 else "official-evaluated breakthrough candidate",
        "decision_reason": "Breakthrough harness never accepts a candidate unless the official evaluator validates it.",
        "raw_output": raw_output,
        "code_snapshot": code_snapshot,
        "code_hash": hash_text(candidate.code),
        "data_hash": hash_packing_data(candidate.data),
        "input_artifacts": {"parent_candidate_id": parent_record.get("candidate_id") if parent_record else None, "source_metadata": source_metadata},
        "output_artifacts": {"code_snapshot": str(code_snapshot), "raw_output": str(raw_output)},
        "diagnostics": candidate.diagnostics,
        "geometry_metrics": metrics,
        "contact_graph": contact,
        "source_metadata": source_metadata,
        "skills_used": ["archive-observability", "packing-repair", "evaluator-feedback"],
    }
    should_archive = eval_result.valid or not archive_public_seed_only_if_valid
    if should_archive:
        archive.add_record(record)
    if eval_result.valid:
        novelty.add(record)
    log_record = {
        "timestamp_utc": utc_stamp(),
        "phase": "official_evaluate",
        "task": task,
        "candidate_id": candidate_id,
        "strategy": candidate.strategy,
        "strategy_family": strategy_family,
        "valid": eval_result.valid,
        "score": eval_result.score,
        "sum_radii": record["sum_radii"],
        "score_improvement": score_improvement,
        "failure_type": eval_result.failure_type,
        "archived": should_archive,
        "code_snapshot": str(code_snapshot),
        "raw_output": str(raw_output),
        "contact_graph": contact,
        "source_metadata": source_metadata,
    }
    _append_jsonl(log_path, log_record)
    return {
        "candidate_id": candidate_id,
        "strategy": candidate.strategy,
        "valid": eval_result.valid,
        "score": eval_result.score,
        "sum_radii": record["sum_radii"],
        "failure_type": eval_result.failure_type,
        "improved": eval_result.valid and score_improvement > 0,
        "archived": should_archive,
        "contact_graph_hash": contact.get("contact_graph_hash"),
        "active_boundary_pattern": contact.get("active_boundary_pattern"),
        "source_metadata": source_metadata,
    }


def _select_parent(task: str, parent_pool: List[PackingData], novelty: NoveltyArchive,
                   fallback: PackingData, batch_index: int) -> PackingData:
    elites = novelty.elites(task)
    if elites and batch_index % 3 == 1:
        data = _packing_from_code_snapshot(Path(str(elites[batch_index % len(elites)].get("code_snapshot"))), task)
        if data is not None:
            return data
    if parent_pool:
        return parent_pool[batch_index % len(parent_pool)]
    return fallback


def _load_current_solution(repo_root: Path, task: str) -> PackingData:
    path = repo_root / ("task_A" if task == "A" else "task_B") / "solution.py"
    data = _packing_from_code_snapshot(path, task)
    if data is None:
        raise RuntimeError(f"Could not load current Task {task} solution from {path}")
    return data


def _packing_from_code_snapshot(path: Path, task: str) -> Optional[PackingData]:
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_breakthrough_{task}_{abs(hash(str(path)))}", str(path))
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


def _compact_best(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
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


def write_breakthrough_report(repo_root: Path, summary: Dict[str, Any],
                              archive: ArchiveManager) -> Path:
    path = repo_root / "submission" / "breakthrough_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    log_rows = _breakthrough_log_rows(repo_root)
    official_rows = [row for row in log_rows if row.get("phase") == "official_evaluate"]
    lines = [
        "# Score Breakthrough Search Report",
        "",
        "The breakthrough harness searches near public benchmark frontier seeds and current best candidates. It does not modify official evaluators, and final `solution.py` remains static, reproducible, and network-free.",
        "",
        "## Public Frontier Seeds",
        "",
        "| Source | Availability | Raw objective claim | Local artifacts |",
        "|---|---|---|---|",
    ]
    for source in (summary.get("frontier_sources") or {}).values():
        lines.append(
            f"| {source.get('name')} | `{source.get('availability')}` | {source.get('raw_objective_claim')} | `{source.get('local_artifacts')}` |"
        )
    lines.extend(["", "## Task Results", ""])
    for task in ("A", "B"):
        task_summary = (summary.get("tasks") or {}).get(task) or {}
        best = task_summary.get("best_after") or {}
        target = float(task_summary.get("target_score") or 1.0)
        score = float(best.get("score") or 0.0)
        gap = max(0.0, target - score)
        lines.extend(
            [
                f"### Task {task}",
                "",
                f"- Best candidate: `{best.get('candidate_id')}`",
                f"- Best sum_radii: `{float(best.get('sum_radii') or 0.0):.15f}`",
                f"- Best score: `{score:.15f}`",
                f"- Gap to denominator score {target:.6f}: `{gap:.15g}`",
                f"- Exceeded denominator: `{score > target}`",
                f"- Generated candidates: `{task_summary.get('generated_count')}`",
                f"- Official evaluated candidates: `{task_summary.get('official_evaluated_count')}`",
                f"- Valid official candidates: `{task_summary.get('valid_count')}`",
                f"- Improvements: `{task_summary.get('improvements')}`",
                "",
            ]
        )
        verified = task_summary.get("frontier_verified") or []
        if verified:
            lines.append("| Verified public seed | Score | Sum radii | Contact graph | Boundary pattern |")
            lines.append("|---|---:|---:|---|---|")
            for item in verified:
                lines.append(
                    f"| `{item.get('candidate_id')}` | {float(item.get('score') or 0.0):.12f} | "
                    f"{float(item.get('sum_radii') or 0.0):.12f} | "
                    f"`{item.get('contact_graph_hash')}` | `{item.get('active_boundary_pattern')}` |"
                )
            lines.append("")
    lines.extend(_strategy_contribution_lines(official_rows))
    lines.extend(_contact_attempt_lines(official_rows))
    lines.extend(
        [
            "## Contact Graph Refinement Evidence",
            "",
            "- Refinement strategy: `contact_graph_feasibility_refine`.",
            "- Deltas attempted: `[1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6]` across batches.",
            "- Each official-evaluated candidate has a code snapshot, raw evaluator output, contact graph hash, active boundary pattern, and failure type in `agent/archive/metrics/breakthrough_log.jsonl`.",
            "- MAP-Elites-lite buckets are stored in `agent/archive/metrics/novelty_archive.json`.",
            "",
            "## Why Evaluators and Final Solutions Stay Safe",
            "",
            "- Official `evaluate.py`, `evaluate_all.py`, and task descriptions are not modified.",
            "- Webpage claimed scores are metadata only; only local official evaluator output validates a candidate.",
            "- Current best is not overwritten unless the official evaluator returns a higher score.",
            "- Final `solution.py` files contain static NumPy arrays and no network calls.",
            "- API keys are not written to logs or archive artifacts.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _breakthrough_log_rows(repo_root: Path) -> List[Dict[str, Any]]:
    path = repo_root / "agent" / "archive" / "metrics" / "breakthrough_log.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _strategy_contribution_lines(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["## Strategy Contribution", "", "No official breakthrough evaluations were logged.", ""]
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        strategy = str(row.get("strategy_family") or row.get("strategy") or "unknown")
        item = grouped.setdefault(strategy, {"attempts": 0, "valid": 0, "best_score": 0.0, "best_sum": 0.0, "failures": {}})
        item["attempts"] += 1
        if row.get("valid"):
            item["valid"] += 1
        if float(row.get("score") or 0.0) > float(item["best_score"]):
            item["best_score"] = float(row.get("score") or 0.0)
            item["best_sum"] = float(row.get("sum_radii") or 0.0)
        failure = str(row.get("failure_type") or "none")
        item["failures"][failure] = item["failures"].get(failure, 0) + 1
    lines = [
        "## Strategy Contribution",
        "",
        "| Strategy family | Official evals | Valid | Best score | Best sum radii | Failure counts |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for strategy, item in sorted(grouped.items(), key=lambda pair: float(pair[1]["best_score"]), reverse=True):
        lines.append(
            f"| `{strategy}` | {item['attempts']} | {item['valid']} | "
            f"{float(item['best_score']):.12f} | {float(item['best_sum']):.12f} | `{item['failures']}` |"
        )
    lines.append("")
    return lines


def _contact_attempt_lines(rows: List[Dict[str, Any]]) -> List[str]:
    contact_rows = [row for row in rows if row.get("strategy_family") == "contact_graph_refinement"]
    if not contact_rows:
        return ["## Contact Graph Attempts", "", "No contact graph refinement candidates were officially evaluated.", ""]
    valid_rows = [row for row in contact_rows if row.get("valid")]
    invalid_rows = [row for row in contact_rows if not row.get("valid")]
    top_rows = sorted(valid_rows, key=lambda row: float(row.get("score") or 0.0), reverse=True)[:8]
    lines = [
        "## Contact Graph Attempts",
        "",
        f"- Official contact graph evaluations: `{len(contact_rows)}`",
        f"- Valid contact graph evaluations: `{len(valid_rows)}`",
        f"- Invalid contact graph evaluations: `{len(invalid_rows)}`",
        "",
        "| Candidate | Task | Score | Sum radii | Contact graph | Boundary pattern |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in top_rows:
        contact = row.get("contact_graph") or {}
        lines.append(
            f"| `{row.get('candidate_id')}` | {row.get('task')} | "
            f"{float(row.get('score') or 0.0):.12f} | "
            f"{float(row.get('sum_radii') or 0.0):.12f} | "
            f"`{contact.get('contact_graph_hash')}` | `{contact.get('active_boundary_pattern')}` |"
        )
    lines.append("")
    return lines


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_json(record), sort_keys=True) + "\n")
