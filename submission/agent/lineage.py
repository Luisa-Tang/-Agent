"""Candidate lineage DAG utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from archive import sanitize_json


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def hash_packing_data(data) -> str:
    payload = {
        "task": data.task,
        "centers": np.asarray(data.centers, dtype=float).round(16).tolist(),
        "radii": np.asarray(data.radii, dtype=float).round(16).tolist(),
        "width": float(data.width) if data.width is not None else None,
        "height": float(data.height) if data.height is not None else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def candidate_node(record: Dict[str, Any]) -> Dict[str, Any]:
    code_path = Path(str(record.get("code_snapshot"))) if record.get("code_snapshot") else None
    code_hash = record.get("code_hash") or (hash_file(code_path) if code_path else None)
    return {
        "candidate_id": record.get("candidate_id"),
        "parent_candidate_id": record.get("parent_candidate_id"),
        "strategy": record.get("strategy"),
        "input_artifacts": record.get("input_artifacts") or {
            "parent_candidate_id": record.get("parent_candidate_id"),
            "source_metadata": record.get("source_metadata"),
        },
        "output_artifacts": record.get("output_artifacts") or {
            "code_snapshot": record.get("code_snapshot"),
            "raw_output": record.get("raw_output"),
        },
        "code_hash": code_hash,
        "data_hash": record.get("data_hash"),
        "official_score": record.get("score"),
        "decision_reason": record.get("decision_reason") or record.get("decision"),
    }


def best_lineage(records: Iterable[Dict[str, Any]], task: str) -> Dict[str, Any]:
    task_records = [record for record in records if str(record.get("task")).upper() == task.upper()]
    valid_records = [record for record in task_records if record.get("valid")]
    best = max(valid_records, key=lambda item: float(item.get("score") or 0.0), default=None)
    nodes = [candidate_node(record) for record in task_records]
    edges = [
        {"from": record.get("parent_candidate_id"), "to": record.get("candidate_id")}
        for record in task_records
        if record.get("parent_candidate_id")
    ]
    by_id = {record.get("candidate_id"): record for record in task_records}
    chain: List[Dict[str, Any]] = []
    current = best
    seen = set()
    while current and current.get("candidate_id") not in seen:
        seen.add(current.get("candidate_id"))
        chain.append(candidate_node(current))
        current = by_id.get(current.get("parent_candidate_id"))
    chain.reverse()
    return {
        "task": task.upper(),
        "best_candidate_id": best.get("candidate_id") if best else None,
        "best_score": float(best.get("score") or 0.0) if best else 0.0,
        "nodes": nodes,
        "edges": edges,
        "best_chain": chain,
        "replay_hint": "Run the code_snapshot for any node through the corresponding official evaluate.py.",
    }


def write_best_lineages(repo_root: Path, records: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    out_dir = Path(repo_root) / "agent" / "archive" / "lineage"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for task in ("A", "B"):
        path = out_dir / f"task_{task}_best_lineage.json"
        path.write_text(
            json.dumps(sanitize_json(best_lineage(records, task)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths[task] = str(path)
    return paths
