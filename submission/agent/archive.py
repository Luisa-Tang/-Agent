"""Archive management for generated candidates and evaluator outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ArchiveManager:
    def __init__(self, repo_root: Path, run_id: str):
        self.repo_root = Path(repo_root).resolve()
        self.run_id = run_id
        self.root = self.repo_root / "agent_runs" / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_jsonl = self.root / "archive.jsonl"
        self.metrics_root = self.repo_root / "agent" / "archive" / "metrics"
        self.metrics_root.mkdir(parents=True, exist_ok=True)
        self.metrics_run_log = self.metrics_root / "run_log.jsonl"
        self.metrics_run_log.write_text("", encoding="utf-8")
        self.records: List[Dict[str, Any]] = []
        self.best: Dict[str, Dict[str, Any]] = {}
        for task in ("A", "B"):
            (self.root / "candidates" / f"task_{task}").mkdir(parents=True, exist_ok=True)
            (self.root / "outputs" / f"task_{task}").mkdir(parents=True, exist_ok=True)

    def make_candidate_id(self, task: str, iteration: int, strategy: str) -> str:
        safe_strategy = "".join(ch if ch.isalnum() else "_" for ch in strategy).strip("_")
        return f"{task.upper()}_{iteration:03d}_{safe_strategy}"

    def save_candidate_code(self, task: str, candidate_id: str, code: str) -> Path:
        path = self.root / "candidates" / f"task_{task.upper()}" / f"{candidate_id}.py"
        path.write_text(code, encoding="utf-8")
        return path

    def save_raw_output(self, task: str, candidate_id: str, raw_output: str) -> Path:
        path = self.root / "outputs" / f"task_{task.upper()}" / f"{candidate_id}.txt"
        path.write_text(raw_output, encoding="utf-8")
        return path

    def add_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = sanitize_json(record)
        self.records.append(record)
        with self.archive_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        with self.metrics_run_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        if record.get("valid"):
            task = str(record["task"]).upper()
            current = self.best.get(task)
            if current is None or float(record.get("score") or 0.0) > float(current.get("score") or 0.0):
                self.best[task] = record
        return record

    def best_record(self, task: str) -> Optional[Dict[str, Any]]:
        return self.best.get(task.upper())

    def export_best_code(self, task: str, destination: Path) -> Optional[Path]:
        record = self.best_record(task)
        if not record:
            return None
        src = Path(record["code_snapshot"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
        return destination

    def records_for_task(self, task: str) -> List[Dict[str, Any]]:
        task = task.upper()
        return [r for r in self.records if str(r.get("task")).upper() == task]

    def strategy_stats(self, task: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        records = self.records if task is None else self.records_for_task(task)
        stats: Dict[str, Dict[str, Any]] = {}
        for record in records:
            strategy = str(record.get("strategy") or "unknown")
            item = stats.setdefault(
                strategy,
                {
                    "attempts": 0,
                    "valid_attempts": 0,
                    "best_score": 0.0,
                    "score_improvements": [],
                },
            )
            item["attempts"] += 1
            score = float(record.get("score") or 0.0)
            if record.get("valid"):
                item["valid_attempts"] += 1
            item["best_score"] = max(float(item["best_score"]), score)
            item["score_improvements"].append(float(record.get("score_improvement") or 0.0))
        for item in stats.values():
            attempts = max(1, int(item["attempts"]))
            improvements = item.pop("score_improvements")
            item["validity_rate"] = float(item["valid_attempts"]) / attempts
            item["average_score_improvement"] = float(np.mean(improvements)) if improvements else 0.0
        return sanitize_json(stats)

    def write_summary(self) -> Path:
        path = self.root / "summary.json"
        payload = {
            "run_id": self.run_id,
            "best": self.best,
            "num_records": len(self.records),
            "archive_jsonl": str(self.archive_jsonl),
            "strategy_stats": self.strategy_stats(),
            "strategy_stats_by_task": {task: self.strategy_stats(task) for task in ("A", "B")},
        }
        path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True), encoding="utf-8")
        return path


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj
