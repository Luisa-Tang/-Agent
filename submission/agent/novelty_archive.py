"""MAP-Elites-lite novelty archive for breakthrough search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from archive import sanitize_json


@dataclass
class NoveltyArchive:
    repo_root: Path
    buckets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def bucket_key(self, record: Dict[str, Any]) -> str:
        metrics = record.get("geometry_metrics") or {}
        contact = record.get("contact_graph") or {}
        task = str(record.get("task") or "unknown")
        score = float(record.get("score") or 0.0)
        min_pair = float(metrics.get("min_pairwise_margin") or contact.get("min_pairwise_margin") or 0.0)
        min_boundary = float(metrics.get("min_boundary_margin") or contact.get("min_boundary_margin") or 0.0)
        width = record.get("width")
        height = record.get("height")
        aspect = ""
        if task == "A" and width and height:
            aspect = f"aspect={_band(float(width) / max(float(height), 1e-12), [0.7, 0.9, 1.1, 1.3])}"
        else:
            aspect = "aspect=unit"
        parts = [
            f"task={task}",
            f"family={record.get('strategy_family') or record.get('strategy') or 'unknown'}",
            f"cg={str(contact.get('contact_graph_hash') or 'none')[:8]}",
            f"boundary={contact.get('active_boundary_pattern') or 'none'}",
            f"score={_score_band(score)}",
            f"safety={_safety_band(min(min_pair, min_boundary))}",
            aspect,
        ]
        return "|".join(parts)

    def add(self, record: Dict[str, Any]) -> Tuple[str, bool]:
        key = self.bucket_key(record)
        current = self.buckets.get(key)
        if current is None or float(record.get("score") or 0.0) > float(current.get("score") or 0.0):
            compact = {
                "bucket_key": key,
                "task": record.get("task"),
                "candidate_id": record.get("candidate_id"),
                "parent_candidate_id": record.get("parent_candidate_id"),
                "strategy": record.get("strategy"),
                "strategy_family": record.get("strategy_family") or record.get("strategy"),
                "score": record.get("score"),
                "sum_radii": record.get("sum_radii"),
                "width": record.get("width"),
                "height": record.get("height"),
                "valid": record.get("valid"),
                "failure_type": record.get("failure_type"),
                "code_snapshot": str(record.get("code_snapshot") or ""),
                "contact_graph": record.get("contact_graph") or {},
                "geometry_metrics": record.get("geometry_metrics") or {},
            }
            self.buckets[key] = sanitize_json(compact)
            return key, True
        return key, False

    def add_many(self, records: Iterable[Dict[str, Any]]) -> None:
        for record in records:
            if record.get("valid"):
                self.add(record)

    def elites(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        values = list(self.buckets.values())
        if task:
            values = [item for item in values if str(item.get("task")).upper() == task.upper()]
        return sorted(values, key=lambda item: float(item.get("score") or 0.0), reverse=True)

    def write(self) -> Path:
        path = self.repo_root / "agent" / "archive" / "metrics" / "novelty_archive.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bucket_count": len(self.buckets),
            "buckets": self.buckets,
            "top_elites": self.elites()[:20],
        }
        path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _score_band(score: float) -> str:
    if score >= 1.0:
        return ">=1.0"
    if score >= 0.99999:
        return "0.99999-1.0"
    if score >= 0.9999:
        return "0.9999-0.99999"
    if score >= 0.99:
        return "0.99-0.9999"
    if score >= 0.9:
        return "0.9-0.99"
    return "<0.9"


def _safety_band(value: float) -> str:
    if value < -1e-7:
        return "violated"
    if value < 1e-10:
        return "tight"
    if value < 1e-7:
        return "near"
    return "slack"


def _band(value: float, cuts: List[float]) -> str:
    lo = "-inf"
    for cut in cuts:
        if value < cut:
            return f"{lo}-{cut:g}"
        lo = f"{cut:g}"
    return f"{lo}-inf"

