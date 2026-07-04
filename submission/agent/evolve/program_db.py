"""Program database for the optional GeoEvolve-lite harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ProgramRecord:
    program_id: str
    parent_program_id: Optional[str]
    task: str
    operator: str
    code_path: str
    code_hash: str
    block_hashes: Dict[str, str] = field(default_factory=dict)
    blocks_used: List[str] = field(default_factory=list)
    block_types_changed: List[str] = field(default_factory=list)
    operator_name: str = ""
    score: float = 0.0
    sum_radii: float = 0.0
    valid: bool = False
    official_valid: bool = False
    official_score: float = 0.0
    official_eval_path: Optional[str] = None
    contact_graph_hash: Optional[str] = None
    boundary_pattern: Optional[str] = None
    contact_graph_changed: bool = False
    boundary_pattern_changed: bool = False
    centers_rmsd_to_parent: float = 0.0
    sorted_radii_l2_to_parent: float = 0.0
    score_delta: float = 0.0
    cascade_stage_reached: str = "created"
    island_name: str = "safe_polish"
    raw_valid: bool = True
    repair_attempted: bool = False
    repair_success: bool = False
    max_violation_before_repair: float = 0.0
    max_violation_after_repair: float = 0.0
    contact_graph_edit_distance: int = 0
    boundary_pattern_edit_distance: int = 0
    small_circle_reassigned: bool = False
    aspect_ratio_bucket_changed: bool = False
    novelty_score: float = 0.0
    strategy_family: str = "self_evolve"
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgramDatabase:
    def __init__(self, repo_root: Path, run_id: str):
        self.repo_root = Path(repo_root)
        self.run_id = run_id
        self.root = self.repo_root / "agent" / "archive" / "evolve"
        self.program_root = self.root / "programs" / run_id
        self.prompt_root = self.root / "prompts" / run_id
        self.program_root.mkdir(parents=True, exist_ok=True)
        self.prompt_root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "program_db.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.records: List[ProgramRecord] = []

    def next_program_id(self, task: str) -> str:
        return f"{task.upper()}_P_{len(self.records) + 1:05d}"

    def write_program(self, program_id: str, code: str) -> Path:
        path = self.program_root / f"{program_id}.py"
        path.write_text(code, encoding="utf-8")
        return path

    def add(self, record: ProgramRecord) -> ProgramRecord:
        if not record.created_at:
            record.created_at = utc_now()
        self.records.append(record)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return record

    def update(self, program_id: str, **fields: Any) -> Optional[ProgramRecord]:
        record = self.get(program_id)
        if record is None:
            return None
        for key, value in fields.items():
            setattr(record, key, value)
        self.rewrite()
        return record

    def get(self, program_id: str) -> Optional[ProgramRecord]:
        for record in self.records:
            if record.program_id == program_id:
                return record
        return None

    def by_task(self, task: str) -> List[ProgramRecord]:
        task = task.upper()
        return [record for record in self.records if record.task.upper() == task]

    def best(self, task: Optional[str] = None) -> Optional[ProgramRecord]:
        records = self.records if task is None else self.by_task(task)
        valid = [record for record in records if record.valid]
        return max(valid, key=lambda item: float(item.score), default=None)

    def diverse_elites(self, task: str, limit: int = 4) -> List[ProgramRecord]:
        seen = set()
        elites = []
        for record in sorted(self.by_task(task), key=lambda item: (item.valid, item.score, item.novelty_score), reverse=True):
            key = (record.contact_graph_hash, record.boundary_pattern, record.strategy_family)
            if key in seen:
                continue
            seen.add(key)
            elites.append(record)
            if len(elites) >= limit:
                break
        return elites

    def rewrite(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def write_tree(self) -> Path:
        path = self.root / "program_tree.json"
        nodes = [asdict(record) for record in self.records]
        edges = [
            {"from": record.parent_program_id, "to": record.program_id}
            for record in self.records
            if record.parent_program_id
        ]
        path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_block_metrics(self) -> Dict[str, Path]:
        jsonl_path = self.root / "block_metrics.jsonl"
        json_path = self.root / "block_metrics.json"
        rows = [self._block_metric_row(record) for record in self.records]
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        aggregate: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            for block_name in row.get("block_types_changed") or ["seed"]:
                item = aggregate.setdefault(
                    block_name,
                    {
                        "attempts": 0,
                        "official_valid": 0,
                        "best_delta": 0.0,
                        "mean_contact_graph_changed": 0.0,
                        "mean_boundary_pattern_changed": 0.0,
                        "mean_centers_rmsd": 0.0,
                        "repair_attempted": 0,
                        "repair_success": 0,
                        "small_circle_reassigned": 0,
                        "accepted_improvements": 0,
                    },
                )
                item["attempts"] += 1
                if row.get("official_valid"):
                    item["official_valid"] += 1
                delta = float(row.get("score_delta") or 0.0)
                item["best_delta"] = max(float(item["best_delta"]), delta)
                item["mean_contact_graph_changed"] += 1.0 if row.get("contact_graph_changed") else 0.0
                item["mean_boundary_pattern_changed"] += 1.0 if row.get("boundary_pattern_changed") else 0.0
                item["mean_centers_rmsd"] += float(row.get("centers_rmsd_to_parent") or 0.0)
                if row.get("repair_attempted"):
                    item["repair_attempted"] += 1
                if row.get("repair_success"):
                    item["repair_success"] += 1
                if row.get("small_circle_reassigned"):
                    item["small_circle_reassigned"] += 1
                if delta > 0.0 and row.get("official_valid"):
                    item["accepted_improvements"] += 1
        for item in aggregate.values():
            attempts = max(1, int(item["attempts"]))
            item["valid_rate"] = float(item["official_valid"]) / attempts
            item["mean_contact_graph_changed"] = float(item["mean_contact_graph_changed"]) / attempts
            item["mean_boundary_pattern_changed"] = float(item["mean_boundary_pattern_changed"]) / attempts
            item["mean_centers_rmsd"] = float(item["mean_centers_rmsd"]) / attempts
        json_path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "program_count": len(rows),
                    "block_stats": aggregate,
                    "programs": rows,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        return {"json": json_path, "jsonl": jsonl_path}

    def _block_metric_row(self, record: ProgramRecord) -> Dict[str, Any]:
        return {
            "program_id": record.program_id,
            "parent_program_id": record.parent_program_id,
            "task": record.task,
            "operator": record.operator,
            "operator_name": record.operator_name or record.operator,
            "blocks_used": record.blocks_used,
            "block_hashes": record.block_hashes,
            "block_types_changed": record.block_types_changed,
            "contact_graph_hash": record.contact_graph_hash,
            "boundary_pattern": record.boundary_pattern,
            "contact_graph_changed": record.contact_graph_changed,
            "boundary_pattern_changed": record.boundary_pattern_changed,
            "centers_rmsd_to_parent": record.centers_rmsd_to_parent,
            "sorted_radii_l2_to_parent": record.sorted_radii_l2_to_parent,
            "score_delta": record.score_delta,
            "official_valid": record.official_valid,
            "official_score": record.official_score,
            "cascade_stage_reached": record.cascade_stage_reached,
            "island_name": record.island_name,
            "raw_valid": record.raw_valid,
            "repair_attempted": record.repair_attempted,
            "repair_success": record.repair_success,
            "max_violation_before_repair": record.max_violation_before_repair,
            "max_violation_after_repair": record.max_violation_after_repair,
            "contact_graph_edit_distance": record.contact_graph_edit_distance,
            "boundary_pattern_edit_distance": record.boundary_pattern_edit_distance,
            "small_circle_reassigned": record.small_circle_reassigned,
            "aspect_ratio_bucket_changed": record.aspect_ratio_bucket_changed,
            "valid": record.valid,
            "score": record.score,
            "sum_radii": record.sum_radii,
            "novelty_score": record.novelty_score,
            "failure_type": (record.metadata or {}).get("failure_type"),
        }


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
