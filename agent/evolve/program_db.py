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
    score: float = 0.0
    sum_radii: float = 0.0
    valid: bool = False
    official_eval_path: Optional[str] = None
    contact_graph_hash: Optional[str] = None
    boundary_pattern: Optional[str] = None
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


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
