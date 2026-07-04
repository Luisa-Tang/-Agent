"""Build static demo data from final solutions and Agent logs.

The script intentionally reads, but does not modify, official evaluator files.
It calls the exported solution entrypoints directly so the visualization reflects
the same final artifacts that are submitted for evaluation.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "submission" / "demo" / "demo_data.json"

TASKS = {
    "A": {
        "label": "Task A",
        "n": 21,
        "target": 2.365840,
        "solution": REPO_ROOT / "task_A" / "solution.py",
        "export_path": "submission/task_A/solution.py",
        "default_width": None,
        "default_height": None,
    },
    "B": {
        "label": "Task B",
        "n": 26,
        "target": 2.635990,
        "solution": REPO_ROOT / "task_B" / "solution.py",
        "export_path": "submission/task_B/solution.py",
        "default_width": 1.0,
        "default_height": 1.0,
    },
}


def main() -> int:
    data = build_demo_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


def build_demo_data() -> Dict[str, Any]:
    final_tasks = {task: load_final_solution(task, spec) for task, spec in TASKS.items()}
    records, log_sources = load_agent_records()
    records = normalize_records(records)

    trajectory_by_task = {
        task: [record for record in records if record.get("task") == task]
        for task in TASKS
    }
    best_candidates = {
        task: best_record(trajectory_by_task[task])
        for task in TASKS
    }
    lineage_by_task = {
        task: build_lineage(trajectory_by_task[task], best_candidates[task])
        for task in TASKS
    }

    score_a = float(final_tasks["A"]["score"])
    score_b = float(final_tasks["B"]["score"])

    return sanitize_json(
        {
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "sources": {
                "solutions": {
                    "A": "task_A/solution.py",
                    "B": "task_B/solution.py",
                },
                "logs": log_sources,
                "notes": [
                    "final geometry is computed by calling run_packing directly",
                    "scores use the same fixed target denominators as official evaluate.py",
                ],
                "langgraph": {
                    "mermaid": "submission/demo/agent_graph.mmd",
                    "text": "submission/demo/agent_graph.txt",
                    "node_log": "agent/archive/metrics/langgraph_run_log.jsonl",
                },
            },
            "tasks": final_tasks,
            "combined_score": (score_a + score_b) / 2.0,
            "trajectory": records,
            "trajectory_by_task": trajectory_by_task,
            "best_candidates": best_candidates,
            "lineage": {
                "A": lineage_by_task["A"],
                "B": lineage_by_task["B"],
            },
            "best_lineage": load_best_lineage_artifacts(),
            "strategy_stats": compute_strategy_stats(records),
            "strategy_stats_by_task": {
                task: compute_strategy_stats(trajectory_by_task[task])
                for task in TASKS
            },
            "strategy_portfolio": load_strategy_portfolio(),
            "safety_guard": load_safety_guard_status(),
            "human_agent_division": load_human_agent_division_summary(),
            "failure_stats": compute_failure_stats(records),
            "failure_stats_by_task": {
                task: compute_failure_stats(trajectory_by_task[task])
                for task in TASKS
            },
            "submission_tree": [
                "submission/",
                "  task_A/solution.py",
                "  task_B/solution.py",
                "  agent/run_archive.jsonl",
                "  agent/run_summary.json",
                "  report.md",
                "  demo/index.html",
                "  demo/demo_data.json",
                "  demo/agent_graph.mmd",
                "  demo/agent_graph.txt",
                "  demo/assets/styles.css",
                "  demo/assets/app.js",
                "  demo/demo_readme.md",
            ],
        }
    )


def load_final_solution(task: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    module = import_solution(spec["solution"], f"_demo_solution_{task.lower()}")
    result = module.run_packing(int(spec["n"]))
    if not isinstance(result, tuple):
        raise TypeError(f"{spec['solution']} run_packing must return a tuple")

    if task == "A":
        if len(result) != 4:
            raise ValueError("Task A run_packing must return centers, radii, width, height")
        centers, radii, width, height = result
    else:
        if len(result) != 3:
            raise ValueError("Task B run_packing must return centers, radii, sum_radii")
        centers, radii, _sum_radii = result
        width, height = float(spec["default_width"]), float(spec["default_height"])

    centers_arr = np.asarray(centers, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    width = float(width)
    height = float(height)
    sum_radii = float(np.sum(radii_arr))
    score = sum_radii / float(spec["target"])
    safety = geometry_safety(centers_arr, radii_arr, width, height)

    return {
        "label": spec["label"],
        "n": int(spec["n"]),
        "target": float(spec["target"]),
        "solution_path": str(spec["solution"].relative_to(REPO_ROOT)),
        "export_path": spec["export_path"],
        "width": width,
        "height": height,
        "sum_radii": sum_radii,
        "score": score,
        "valid_shape": centers_arr.shape == (int(spec["n"]), 2) and radii_arr.shape == (int(spec["n"]),),
        "min_pairwise_margin": safety["min_pairwise_margin"],
        "min_boundary_margin": safety["min_boundary_margin"],
        "centers": centers_arr.tolist(),
        "radii": radii_arr.tolist(),
    }


def import_solution(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_packing"):
        raise AttributeError(f"{path} has no run_packing")
    return module


def geometry_safety(centers: np.ndarray, radii: np.ndarray, width: float, height: float) -> Dict[str, float]:
    boundary = np.minimum.reduce(
        [
            centers[:, 0] - radii,
            width - centers[:, 0] - radii,
            centers[:, 1] - radii,
            height - centers[:, 1] - radii,
        ]
    )
    min_pairwise = math.inf
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            min_pairwise = min(min_pairwise, dist - float(radii[i] + radii[j]))
    if not math.isfinite(min_pairwise):
        min_pairwise = 0.0
    return {
        "min_pairwise_margin": float(min_pairwise),
        "min_boundary_margin": float(np.min(boundary)) if len(boundary) else 0.0,
    }


def load_agent_records() -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    primary = REPO_ROOT / "agent" / "archive" / "metrics" / "run_log.jsonl"
    if primary.exists():
        return load_jsonl_records(primary), [{"path": rel(primary), "type": "jsonl"}]

    legacy_sources = [
        ("A", REPO_ROOT / "task_A" / "run_log_a.log"),
        ("B", REPO_ROOT / "task_B" / "run_log_b.log"),
    ]
    records: List[Dict[str, Any]] = []
    sources: List[Dict[str, str]] = []
    for task, path in legacy_sources:
        if path.exists():
            records.extend(parse_legacy_log(task, path))
            sources.append({"path": rel(path), "type": "legacy_text_log"})
    if records:
        return records, sources

    structured_candidates = [
        REPO_ROOT / "submission" / "agent" / "run_archive.jsonl",
        latest_agent_run_archive(),
    ]
    for path in structured_candidates:
        if path and path.exists():
            records = load_jsonl_records(path)
            if records:
                return records, [{"path": rel(path), "type": "jsonl_fallback"}]
    return records, sources


def latest_agent_run_archive() -> Optional[Path]:
    root = REPO_ROOT / "agent_runs"
    if not root.exists():
        return None
    paths = sorted(root.glob("*/archive.jsonl"), key=lambda item: item.parent.name, reverse=True)
    return paths[0] if paths else None


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("candidate_id"):
                records.append(record)
    return records


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_best_lineage_artifacts() -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for task in TASKS:
        path = REPO_ROOT / "agent" / "archive" / "lineage" / f"task_{task}_best_lineage.json"
        lineage = load_json_file(path)
        if lineage:
            payload[task] = {
                "path": rel(path),
                "best_candidate_id": lineage.get("best_candidate_id"),
                "best_score": lineage.get("best_score"),
                "node_count": len(lineage.get("nodes") or []),
                "edge_count": len(lineage.get("edges") or []),
                "best_chain": lineage.get("best_chain") or [],
                "replay_hint": lineage.get("replay_hint"),
            }
        else:
            payload[task] = {
                "path": rel(path),
                "available": False,
            }
    return payload


def load_strategy_portfolio() -> Dict[str, Any]:
    path = REPO_ROOT / "agent" / "archive" / "metrics" / "strategy_portfolio.json"
    payload = load_json_file(path)
    if not payload:
        return {"path": rel(path), "available": False}
    payload["path"] = rel(path)
    return payload


def load_safety_guard_status() -> Dict[str, Any]:
    paths = [
        REPO_ROOT / "submission" / "safety_report.json",
        REPO_ROOT / "agent" / "archive" / "metrics" / "safety_report.json",
    ]
    for path in paths:
        payload = load_json_file(path)
        if payload:
            protected = payload.get("protected_files") or {}
            secret = payload.get("secret_scan") or {}
            return {
                "path": rel(path),
                "passed": payload.get("passed"),
                "protected_files_unchanged": protected.get("unchanged"),
                "api_key_pattern_matches": len(secret.get("matches") or []),
                "final_solutions": payload.get("final_solutions") or {},
            }
    return {"available": False}


def load_human_agent_division_summary() -> Dict[str, Any]:
    path = REPO_ROOT / "submission" / "human_agent_division.json"
    payload = load_json_file(path)
    if not payload:
        return {"path": rel(path), "available": False}
    return {
        "path": rel(path),
        "human_provided_count": len(payload.get("human_provided") or []),
        "agent_completed_count": len(payload.get("agent_completed") or []),
        "best_candidates": payload.get("best_candidates") or {},
        "human_provided": payload.get("human_provided") or [],
        "agent_completed": payload.get("agent_completed") or [],
    }


def parse_legacy_log(task: str, path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"Iteration\s+(\d+)\s+\|\s+candidate\s+(\S+)(.*?)(?=\nIteration\s+\d+\s+\||\Z)", re.S)
    records = []
    for match in pattern.finditer(text):
        iteration = int(match.group(1))
        candidate_id = match.group(2)
        block = match.group(3)
        strategy = first_match(block, r"\n\s+strategy:\s*([^\n]+)")
        if strategy is None:
            strategy = first_match(block, r"think:\s*selected=([^|\n]+)")
        valid_text = first_match(block, r"valid:\s*(True|False)")
        score = parse_float(first_match(block, r"score:\s*([0-9.eE+-]+)"))
        sum_radii = parse_float(first_match(block, r"sum_radii:\s*([0-9.eE+-]+)"))
        failure_type = first_match(block, r"failure_type:\s*([^\n]+)") or "unknown"
        decision = first_match(block, r"decision:\s*([^\n]+)") or ""
        raw_output = first_match(block, r"raw_output:\s*([^\n]+)")
        code_snapshot = first_match(block, r"code_snapshot:\s*([^\n]+)")
        min_pair = parse_float(first_match(block, r"min_pairwise_margin=([0-9.eE+-]+)"))
        min_boundary = parse_float(first_match(block, r"min_boundary_margin=([0-9.eE+-]+)"))
        records.append(
            {
                "task": task,
                "iteration": iteration,
                "candidate_id": candidate_id,
                "strategy": (strategy or "unknown").strip(),
                "valid": valid_text == "True",
                "score": score,
                "sum_radii": sum_radii,
                "failure_type": failure_type.strip(),
                "decision": decision.strip(),
                "raw_output": raw_output,
                "code_snapshot": code_snapshot,
                "geometry_metrics": {
                    "min_pairwise_margin": min_pair,
                    "min_boundary_margin": min_boundary,
                },
            }
        )
    return records


def first_match(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for record in records:
        task = str(record.get("task") or "").upper()
        if task not in TASKS:
            continue
        metrics = record.get("geometry_metrics") or {}
        trace = record.get("trace") or {}
        action = trace.get("action") or {}
        next_observation = trace.get("next_observation") or {}
        compact.append(
            {
                "task": task,
                "iteration": int(record.get("iteration") or 0),
                "candidate_id": record.get("candidate_id"),
                "parent_candidate_id": record.get("parent_candidate_id"),
                "strategy": record.get("strategy") or record.get("local_policy_strategy") or "unknown",
                "valid": bool(record.get("valid")),
                "score": float(record.get("score") or 0.0),
                "sum_radii": float(record.get("sum_radii") or 0.0),
                "score_improvement": float(record.get("score_improvement") or 0.0),
                "width": record.get("width"),
                "height": record.get("height"),
                "failure_type": record.get("failure_type") or "unknown",
                "decision": record.get("decision") or "",
                "skills_used": record.get("skills_used") or action.get("skills_used") or [],
                "optimizer_or_repair": action.get("optimizer_or_repair"),
                "elapsed": next_observation.get("elapsed"),
                "raw_output": path_relative_if_possible(record.get("raw_output")),
                "code_snapshot": path_relative_if_possible(record.get("code_snapshot")),
                "geometry_metrics": {
                    "min_pairwise_margin": parse_float(metrics.get("min_pairwise_margin")),
                    "min_boundary_margin": parse_float(metrics.get("min_boundary_margin")),
                    "width": metrics.get("width"),
                    "height": metrics.get("height"),
                },
            }
        )
    compact.sort(key=lambda item: (item["task"], item["iteration"], str(item["candidate_id"])))
    return compact


def best_record(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid_records = [record for record in records if record.get("valid")]
    if not valid_records:
        return None
    return max(valid_records, key=lambda item: (float(item.get("score") or 0.0), -int(item.get("iteration") or 0)))


def build_lineage(records: List[Dict[str, Any]], best: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not best:
        return []
    by_id = {record.get("candidate_id"): record for record in records if record.get("candidate_id")}
    chain = []
    current = best
    seen = set()
    while current and current.get("candidate_id") not in seen:
        seen.add(current.get("candidate_id"))
        chain.append(current)
        parent_id = current.get("parent_candidate_id")
        current = by_id.get(parent_id)
    chain.reverse()
    if len(chain) > 1:
        return chain

    best_so_far = -math.inf
    improvement_path = []
    for record in sorted(records, key=lambda item: int(item.get("iteration") or 0)):
        score = float(record.get("score") or 0.0)
        if record.get("valid") and score > best_so_far:
            improvement_path.append(record)
            best_so_far = score
    return improvement_path or chain


def compute_strategy_stats(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    improvements: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        strategy = str(record.get("strategy") or "unknown")
        item = grouped.setdefault(
            strategy,
            {
                "attempts": 0,
                "valid_attempts": 0,
                "best_score": 0.0,
                "avg_score": 0.0,
            },
        )
        item["attempts"] += 1
        if record.get("valid"):
            item["valid_attempts"] += 1
        score = float(record.get("score") or 0.0)
        item["best_score"] = max(float(item["best_score"]), score)
        item["avg_score"] += score
        improvements[strategy].append(float(record.get("score_improvement") or 0.0))
    for strategy, item in grouped.items():
        attempts = max(1, int(item["attempts"]))
        item["validity_rate"] = float(item["valid_attempts"]) / attempts
        item["avg_score"] = float(item["avg_score"]) / attempts
        item["average_score_improvement"] = float(np.mean(improvements[strategy])) if improvements[strategy] else 0.0
    return dict(sorted(grouped.items()))


def compute_failure_stats(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(str(record.get("failure_type") or "unknown") for record in records)
    return dict(sorted(counter.items()))


def path_relative_if_possible(value: Any) -> Any:
    if not value:
        return value
    try:
        path = Path(str(value))
        if path.is_absolute():
            return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(value)
    return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
