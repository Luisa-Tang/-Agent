"""Cascade evaluator for self-evolved candidate-generating programs."""

from __future__ import annotations

import importlib.util
import json
import py_compile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from archive import ArchiveManager, sanitize_json
from candidate_generators import solution_code_for
from contact_graph import summarize_contact_graph
from evaluator_adapter import EvaluatorAdapter, EvalResult
from geometry_utils import (
    PackingData,
    clip_centers,
    safety_metrics,
    safety_repair,
    solve_radii_lp,
    validate_packing,
)
from lineage import hash_packing_data, hash_text

try:
    from .novelty_filter import NoveltyDecision, NoveltyFilter
    from .program_db import ProgramDatabase, ProgramRecord
except ImportError:  # pragma: no cover - direct module execution fallback
    from novelty_filter import NoveltyDecision, NoveltyFilter
    from program_db import ProgramDatabase, ProgramRecord


@dataclass
class PreparedCandidate:
    program_record: ProgramRecord
    data: Optional[PackingData] = None
    solution_code: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    contact_graph: Dict[str, Any] = field(default_factory=dict)
    geometry_metrics: Dict[str, Any] = field(default_factory=dict)
    novelty: Optional[NoveltyDecision] = None
    internal_valid: bool = False
    failure_type: str = "none"
    eligible_for_official: bool = False


class CascadeEvaluator:
    def __init__(self, repo_root: Path, adapter: EvaluatorAdapter,
                 archive: ArchiveManager, program_db: ProgramDatabase):
        self.repo_root = Path(repo_root)
        self.adapter = adapter
        self.archive = archive
        self.program_db = program_db
        self.log_path = self.repo_root / "agent" / "archive" / "evolve" / "evolve_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def reset_log(self) -> None:
        self.log_path.write_text("", encoding="utf-8")

    def prepare(self, record: ProgramRecord, parent_data: PackingData,
                context: Dict[str, Any], novelty_filter: NoveltyFilter,
                novelty_threshold: float) -> PreparedCandidate:
        start = time.time()
        prepared = PreparedCandidate(program_record=record)
        e0 = self._syntax_import_check(record)
        if not e0["ok"]:
            prepared.failure_type = e0["failure_type"]
            prepared.diagnostics = e0
            self._update_program_failure(record, prepared, runtime=time.time() - start)
            self._append_log("E0_syntax_import", record, prepared, context)
            return prepared
        self._append_log("E0_syntax_import", record, prepared, context, extra=e0)

        try:
            proposed = self._run_program(record, parent_data, context)
            data, candidate_meta = self._candidate_data(record.task, proposed, parent_data, context)
        except Exception as exc:
            prepared.failure_type = "program_runtime_error"
            prepared.diagnostics = {"exception": repr(exc)}
            self._update_program_failure(record, prepared, runtime=time.time() - start)
            self._append_log("E1_internal_geometry", record, prepared, context)
            return prepared

        valid, message = validate_packing(record.task, data.centers, data.radii, data.width, data.height, tol=1e-8)
        prepared.data = data
        prepared.internal_valid = bool(valid)
        prepared.failure_type = "none" if valid else str(message)
        prepared.geometry_metrics = safety_metrics(data)
        prepared.contact_graph = summarize_contact_graph(
            record.task,
            data.centers,
            data.radii,
            data.width,
            data.height,
            tolerance=float(context.get("contact_tolerance", 5e-8) or 5e-8),
        )
        prepared.diagnostics = {
            "candidate_metadata": candidate_meta,
            "operator_context": _safe_context(context),
            "internal_valid": bool(valid),
            "internal_message": message,
            "sum_radii": data.sum_radii,
            "score_estimate": data.score,
            "stage": "E1",
        }
        self._append_log("E1_internal_geometry", record, prepared, context)
        if not valid:
            self._update_program_failure(record, prepared, runtime=time.time() - start)
            return prepared

        decision = novelty_filter.judge(
            code_hash=record.code_hash,
            contact_hash=str(prepared.contact_graph.get("contact_graph_hash") or "none"),
            boundary_pattern=str(prepared.contact_graph.get("active_boundary_pattern") or "none"),
            strategy_family=record.strategy_family,
            data=data,
            parent_data=parent_data,
        )
        prepared.novelty = decision
        prepared.failure_type = decision.failure_type
        prepared.eligible_for_official = bool(decision.accepted)
        prepared.diagnostics["novelty"] = {
            "accepted": decision.accepted,
            "score": decision.novelty_score,
            "reasons": decision.reasons,
            "threshold": novelty_threshold,
        }
        prepared.solution_code = solution_code_for(
            data,
            strategy=f"self_evolve_{record.operator}",
            diagnostics=prepared.diagnostics,
        )
        self.program_db.update(
            record.program_id,
            sum_radii=data.sum_radii,
            score=data.score,
            contact_graph_hash=prepared.contact_graph.get("contact_graph_hash"),
            boundary_pattern=prepared.contact_graph.get("active_boundary_pattern"),
            novelty_score=decision.novelty_score,
            metadata={**record.metadata, **prepared.diagnostics, "failure_type": prepared.failure_type},
        )
        self._append_log("E2_novelty_quick_score", record, prepared, context)
        return prepared

    def official_evaluate(self, prepared: PreparedCandidate,
                          parent_archive_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        record = prepared.program_record
        if prepared.data is None or prepared.solution_code is None:
            raise ValueError("Prepared candidate is missing data or solution code")
        start = time.time()
        candidate_id = f"{record.task}_evolve_{record.program_id}"
        previous_best = self.archive.best_record(record.task)
        previous_score = float(previous_best.get("score") or 0.0) if previous_best else 0.0
        eval_result = self.adapter.evaluate_task(record.task, prepared.solution_code)
        code_snapshot = self.archive.save_candidate_code(record.task, candidate_id, prepared.solution_code)
        raw_output = self.archive.save_raw_output(record.task, candidate_id, eval_result.raw_output)
        score_improvement = float(eval_result.score) - previous_score
        failure_type = eval_result.failure_type or "none"
        archive_record = {
            "task": record.task,
            "iteration": 99000 + len(self.archive.records_for_task(record.task)),
            "candidate_id": candidate_id,
            "strategy": f"self_evolve_{record.operator}",
            "strategy_family": record.strategy_family,
            "parent_candidate_id": parent_archive_record.get("candidate_id") if parent_archive_record else None,
            "parent_program_id": record.parent_program_id,
            "program_id": record.program_id,
            "valid": eval_result.valid,
            "score": eval_result.score,
            "sum_radii": eval_result.sum_radii if eval_result.sum_radii is not None else prepared.data.sum_radii,
            "width": prepared.data.width if record.task == "A" else None,
            "height": prepared.data.height if record.task == "A" else None,
            "failure_type": failure_type,
            "raw_failure_type": failure_type,
            "score_improvement": score_improvement,
            "decision": "official-valid improvement" if eval_result.valid and score_improvement > 0 else "official-evaluated self-evolution candidate",
            "decision_reason": "Self-evolution candidates can replace the final solution only after official evaluator validation and score improvement.",
            "raw_output": raw_output,
            "code_snapshot": code_snapshot,
            "code_hash": hash_text(prepared.solution_code),
            "data_hash": hash_packing_data(prepared.data),
            "input_artifacts": {
                "program_path": record.code_path,
                "parent_program_id": record.parent_program_id,
            },
            "output_artifacts": {
                "code_snapshot": str(code_snapshot),
                "raw_output": str(raw_output),
                "solution_path": str(self.adapter.solution_path(record.task)),
            },
            "diagnostics": prepared.diagnostics,
            "geometry_metrics": prepared.geometry_metrics,
            "contact_graph": prepared.contact_graph,
            "source_metadata": {
                "source": "GeoEvolve-lite self-evolution harness",
                "inspired_by": ["OpenEvolve", "ShinkaEvolve", "CodeEvolve"],
                "program_id": record.program_id,
                "operator": record.operator,
                "official_evaluator_result": {
                    "valid": eval_result.valid,
                    "score": eval_result.score,
                    "sum_radii": eval_result.sum_radii,
                    "failure_type": failure_type,
                    "returncode": eval_result.returncode,
                    "raw_output_path": str(raw_output),
                    "exact_sum_radii": eval_result.exact_sum_radii,
                    "exact_score": eval_result.exact_score,
                },
            },
            "skills_used": ["archive-observability", "packing-repair", "evaluator-feedback", "static-export"],
        }
        self.archive.add_record(archive_record)
        self.program_db.update(
            record.program_id,
            valid=bool(eval_result.valid),
            score=float(eval_result.score),
            sum_radii=float(archive_record["sum_radii"] or 0.0),
            official_eval_path=str(raw_output),
            contact_graph_hash=prepared.contact_graph.get("contact_graph_hash"),
            boundary_pattern=prepared.contact_graph.get("active_boundary_pattern"),
            novelty_score=float(prepared.novelty.novelty_score if prepared.novelty else 0.0),
            metadata={
                **record.metadata,
                **prepared.diagnostics,
                "failure_type": failure_type,
                "decision_reason": archive_record["decision_reason"],
                "official_candidate_id": candidate_id,
                "official_runtime_seconds": time.time() - start,
            },
        )
        result = {
            "phase": "E3_official_evaluate",
            "candidate_id": candidate_id,
            "program_id": record.program_id,
            "task": record.task,
            "operator": record.operator,
            "valid": eval_result.valid,
            "score": eval_result.score,
            "sum_radii": archive_record["sum_radii"],
            "score_improvement": score_improvement,
            "failure_type": failure_type,
            "raw_output": str(raw_output),
            "code_snapshot": str(code_snapshot),
            "elapsed_seconds": time.time() - start,
        }
        self._append_log("E3_official_evaluate", record, prepared, {}, extra=result)
        return result

    def log_official_skip(self, prepared: PreparedCandidate, reason: str,
                          context: Dict[str, Any]) -> None:
        prepared.failure_type = reason
        prepared.diagnostics["official_skip_reason"] = reason
        self.program_db.update(
            prepared.program_record.program_id,
            metadata={**prepared.program_record.metadata, **prepared.diagnostics, "failure_type": reason},
        )
        self._append_log("E3_official_skipped", prepared.program_record, prepared, context)

    def _syntax_import_check(self, record: ProgramRecord) -> Dict[str, Any]:
        path = Path(record.code_path)
        try:
            py_compile.compile(str(path), doraise=True)
            module = self._import_program(path)
            if not hasattr(module, "propose_candidate"):
                return {"ok": False, "failure_type": "missing_propose_candidate"}
            return {"ok": True, "failure_type": "none"}
        except Exception as exc:
            return {"ok": False, "failure_type": "syntax_or_import_error", "exception": repr(exc)}

    def _run_program(self, record: ProgramRecord, parent_data: PackingData,
                     context: Dict[str, Any]) -> Dict[str, Any]:
        module = self._import_program(Path(record.code_path))
        rng = np.random.default_rng(int(context.get("seed", 0) or 0))
        result = module.propose_candidate(_parent_payload(parent_data), rng, context)
        if not isinstance(result, dict):
            raise TypeError("propose_candidate must return a dict")
        return result

    def _import_program(self, path: Path):
        module_name = f"_geoevolve_{path.stem}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load evolved program {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _candidate_data(self, task: str, proposed: Dict[str, Any],
                        parent_data: PackingData, context: Dict[str, Any]) -> tuple:
        task = task.upper()
        centers = np.asarray(proposed.get("centers"), dtype=float)
        if centers.shape != parent_data.centers.shape:
            raise ValueError(f"proposed centers shape {centers.shape} != {parent_data.centers.shape}")
        margin = float(context.get("lp_margin", 0.0) or 0.0)
        safety = float(context.get("safety", 2e-10) or 2e-10)
        metadata = proposed.get("metadata") if isinstance(proposed.get("metadata"), dict) else {}
        if task == "A":
            width = float(proposed.get("width", parent_data.width))
            width = float(np.clip(width, 0.28, 1.72))
            height = 2.0 - width
            centers = clip_centers(task, centers, width, height, eps=max(safety, 1e-10))
            radii = solve_radii_lp(task, centers, width, height, margin=margin)
            centers, radii = safety_repair(task, centers, radii, width, height, safety=safety)
            return PackingData(task=task, centers=centers, radii=radii, width=width, height=height), metadata
        centers = clip_centers(task, centers, eps=max(safety, 1e-10))
        radii = solve_radii_lp(task, centers, margin=margin)
        centers, radii = safety_repair(task, centers, radii, safety=safety)
        return PackingData(task=task, centers=centers, radii=radii), metadata

    def _update_program_failure(self, record: ProgramRecord, prepared: PreparedCandidate,
                                runtime: float) -> None:
        self.program_db.update(
            record.program_id,
            valid=False,
            score=float(prepared.data.score if prepared.data else 0.0),
            sum_radii=float(prepared.data.sum_radii if prepared.data else 0.0),
            metadata={
                **record.metadata,
                **prepared.diagnostics,
                "failure_type": prepared.failure_type,
                "runtime_seconds": runtime,
            },
        )

    def _append_log(self, phase: str, record: ProgramRecord, prepared: PreparedCandidate,
                    context: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "timestamp_utc": utc_now(),
            "phase": phase,
            "task": record.task,
            "program_id": record.program_id,
            "parent_program_id": record.parent_program_id,
            "operator": record.operator,
            "strategy_family": record.strategy_family,
            "code_path": record.code_path,
            "code_hash": record.code_hash,
            "sum_radii": prepared.data.sum_radii if prepared.data else None,
            "score": prepared.data.score if prepared.data else 0.0,
            "internal_valid": prepared.internal_valid,
            "failure_type": prepared.failure_type,
            "eligible_for_official": prepared.eligible_for_official,
            "novelty_score": prepared.novelty.novelty_score if prepared.novelty else record.novelty_score,
            "novelty": prepared.novelty.reasons if prepared.novelty else None,
            "geometry_metrics": prepared.geometry_metrics,
            "contact_graph": prepared.contact_graph,
            "context": _safe_context(context),
            "diagnostics": prepared.diagnostics,
        }
        if extra:
            payload.update(extra)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitize_json(payload), sort_keys=True) + "\n")


def _parent_payload(data: PackingData) -> Dict[str, Any]:
    payload = {
        "task": data.task,
        "centers": np.asarray(data.centers, dtype=float),
        "radii": np.asarray(data.radii, dtype=float),
        "sum_radii": data.sum_radii,
        "score": data.score,
    }
    if data.task == "A":
        payload["width"] = float(data.width)
        payload["height"] = float(data.height)
    return payload


def _safe_context(context: Dict[str, Any]) -> Dict[str, Any]:
    safe = {}
    for key, value in context.items():
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            safe[key] = "<redacted>"
        elif key == "mate" and isinstance(value, dict):
            safe[key] = {
                "available": True,
                "width": value.get("width"),
                "height": value.get("height"),
                "centers_shape": list(np.asarray(value.get("centers")).shape) if value.get("centers") is not None else None,
            }
        else:
            safe[key] = value
    return safe


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
