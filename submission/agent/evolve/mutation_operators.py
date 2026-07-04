"""Program-level mutation operators for GeoEvolve-lite."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from .program_db import ProgramDatabase, ProgramRecord, code_hash
except ImportError:  # pragma: no cover - direct module execution fallback
    from program_db import ProgramDatabase, ProgramRecord, code_hash


EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"


def template_program_code(task: str) -> str:
    task = task.upper()
    name = "task_a_candidate_program.py" if task == "A" else "task_b_candidate_program.py"
    return (Path(__file__).resolve().parent / "evolve_blocks" / name).read_text(encoding="utf-8")


def create_seed_program(db: ProgramDatabase, task: str, data_metadata: Dict[str, Any]) -> ProgramRecord:
    program_id = db.next_program_id(task)
    code = template_program_code(task)
    path = db.write_program(program_id, code)
    return db.add(
        ProgramRecord(
            program_id=program_id,
            parent_program_id=None,
            task=task.upper(),
            operator="seed_parent_program",
            code_path=str(path),
            code_hash=code_hash(code),
            score=float(data_metadata.get("score") or 0.0),
            sum_radii=float(data_metadata.get("sum_radii") or 0.0),
            valid=True,
            contact_graph_hash=data_metadata.get("contact_graph_hash"),
            boundary_pattern=data_metadata.get("boundary_pattern"),
            novelty_score=1.0,
            strategy_family="seed_parent_program",
            metadata=dict(data_metadata),
        )
    )


def create_child_program(db: ProgramDatabase, task: str, parent_record: ProgramRecord,
                         operator: str, rng: np.random.Generator,
                         context: Dict[str, Any],
                         prompt_path: Optional[Path] = None) -> ProgramRecord:
    parent_code = Path(parent_record.code_path).read_text(encoding="utf-8")
    block = _operator_block(task, operator, rng, context)
    child_code = replace_evolve_block(parent_code, block)
    program_id = db.next_program_id(task)
    child_path = db.write_program(program_id, child_code)
    metadata = {
        "operator_context": _metadata_context(context),
        "prompt_path": str(prompt_path) if prompt_path else None,
        "parent_code_hash": parent_record.code_hash,
        "llm_used": False,
        "llm_fallback_reason": _llm_fallback_reason(operator, context),
        "inspirations": ["OpenEvolve", "ShinkaEvolve", "CodeEvolve"],
        "note": "Deterministic EVOLVE-BLOCK mutation; final solution export still requires official evaluator validity.",
    }
    return db.add(
        ProgramRecord(
            program_id=program_id,
            parent_program_id=parent_record.program_id,
            task=task.upper(),
            operator=operator,
            code_path=str(child_path),
            code_hash=code_hash(child_code),
            strategy_family=_strategy_family(operator),
            metadata=metadata,
        )
    )


def extract_evolve_block(code: str) -> str:
    start = code.index(EVOLVE_START)
    end = code.index(EVOLVE_END, start)
    return code[start + len(EVOLVE_START):end].strip()


def replace_evolve_block(code: str, block: str) -> str:
    start = code.index(EVOLVE_START)
    end = code.index(EVOLVE_END, start) + len(EVOLVE_END)
    replacement = EVOLVE_START + "\n" + block.rstrip() + "\n" + EVOLVE_END
    return code[:start] + replacement + code[end:]


def _operator_block(task: str, operator: str, rng: np.random.Generator,
                    context: Dict[str, Any]) -> str:
    task = task.upper()
    if operator == "solver_switch":
        return _solver_switch_block(task)
    if operator == "contact_threshold_mutation":
        return _contact_threshold_block(task)
    if operator == "program_patch":
        return _program_patch_fallback_block(task)
    if operator == "crossover":
        return _crossover_block(task)
    if operator == "depth_refinement":
        return _depth_refinement_block(task)
    return _parameter_mutation_block(task)


def _parameter_mutation_block(task: str) -> str:
    if task == "A":
        return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    sigma = float(context.get("sigma", 3e-7) or 3e-7)
    width_sigma = float(context.get("width_sigma", sigma) or sigma)
    centers += rng.normal(0.0, sigma, size=centers.shape)
    width = float(np.clip(width + float(rng.normal(0.0, width_sigma)), 0.28, 1.72))
    height = 2.0 - width
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "parameter_mutation", "sigma": sigma, "width_sigma": width_sigma}}'''
    return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    sigma = float(context.get("sigma", 3e-7) or 3e-7)
    centers += rng.normal(0.0, sigma, size=centers.shape)
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "parameter_mutation", "sigma": sigma}}'''


def _solver_switch_block(task: str) -> str:
    if task == "A":
        return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    height = 2.0 - width
    jitter = float(context.get("solver_switch_jitter", 1e-7) or 1e-7)
    direction = centers - np.array([[0.5 * width, 0.5 * height]])
    norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    centers += direction / norm * rng.normal(0.0, jitter, size=(len(centers), 1))
    centers += rng.normal(0.0, jitter * 0.3, size=centers.shape)
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "solver_switch", "solver": context.get("solver", "lp")}}'''
    return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    jitter = float(context.get("solver_switch_jitter", 1e-7) or 1e-7)
    direction = centers - 0.5
    norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    centers += direction / norm * rng.normal(0.0, jitter, size=(len(centers), 1))
    centers += rng.normal(0.0, jitter * 0.3, size=centers.shape)
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "solver_switch", "solver": context.get("solver", "lp")}}'''


def _contact_threshold_block(task: str) -> str:
    if task == "A":
        return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    height = 2.0 - width
    threshold = float(context.get("contact_threshold", 5e-8) or 5e-8)
    scale = float(context.get("contact_scale", 0.35) or 0.35)
    midpoint = np.array([[0.5 * width, 0.5 * height]])
    direction = centers - midpoint
    norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    centers += direction / norm * rng.normal(0.0, threshold * scale, size=(len(centers), 1))
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "contact_threshold_mutation", "contact_threshold": threshold}}'''
    return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    threshold = float(context.get("contact_threshold", 5e-8) or 5e-8)
    scale = float(context.get("contact_scale", 0.35) or 0.35)
    direction = centers - 0.5
    norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    centers += direction / norm * rng.normal(0.0, threshold * scale, size=(len(centers), 1))
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "contact_threshold_mutation", "contact_threshold": threshold}}'''


def _program_patch_fallback_block(task: str) -> str:
    if task == "A":
        return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    height = 2.0 - width
    sigma = float(context.get("sigma", 1e-6) or 1e-6)
    order = np.argsort(-radii)
    for rank, idx in enumerate(order[: max(1, len(order) // 3)]):
        angle = 2.399963229728653 * (rank + 1)
        centers[idx, 0] += sigma * np.cos(angle)
        centers[idx, 1] += sigma * np.sin(angle)
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "program_patch_fallback", "llm_used": False, "sigma": sigma}}'''
    return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    sigma = float(context.get("sigma", 1e-6) or 1e-6)
    order = np.argsort(-radii)
    for rank, idx in enumerate(order[: max(1, len(order) // 3)]):
        angle = 2.399963229728653 * (rank + 1)
        centers[idx, 0] += sigma * np.cos(angle)
        centers[idx, 1] += sigma * np.sin(angle)
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "program_patch_fallback", "llm_used": False, "sigma": sigma}}'''


def _crossover_block(task: str) -> str:
    if task == "A":
        return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    mate = context.get("mate") or {}
    if "centers" in mate:
        mate_centers = np.asarray(mate["centers"], dtype=float)
        if mate_centers.shape == centers.shape:
            mask = (np.arange(len(centers)) % 2) == 1
            centers[mask] = 0.7 * centers[mask] + 0.3 * mate_centers[mask]
    if mate.get("width") is not None:
        width = float(np.clip(0.7 * width + 0.3 * float(mate.get("width")), 0.28, 1.72))
    height = 2.0 - width
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "crossover", "mate_available": bool(mate)}}'''
    return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    mate = context.get("mate") or {}
    if "centers" in mate:
        mate_centers = np.asarray(mate["centers"], dtype=float)
        if mate_centers.shape == centers.shape:
            mask = (np.arange(len(centers)) % 2) == 1
            centers[mask] = 0.7 * centers[mask] + 0.3 * mate_centers[mask]
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "crossover", "mate_available": bool(mate)}}'''


def _depth_refinement_block(task: str) -> str:
    if task == "A":
        return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    width = float(parent.get("width", 1.0) or 1.0)
    height = 2.0 - width
    target_delta = float(context.get("target_delta", 1e-8) or 1e-8)
    midpoint = np.array([[0.5 * width, 0.5 * height]])
    direction = centers - midpoint
    centers += direction * target_delta * 0.08
    width = float(np.clip(width + target_delta * float(context.get("width_direction", 1.0)), 0.28, 1.72))
    height = 2.0 - width
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, width - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, height - eps)
    return {"centers": centers, "radii": radii, "width": width, "height": height,
            "metadata": {"strategy_family": "depth_refinement", "target_delta": target_delta}}'''
    return '''def propose_candidate(parent, rng, context):
    import numpy as np

    centers = np.asarray(parent["centers"], dtype=float).copy()
    radii = np.asarray(parent["radii"], dtype=float).copy()
    target_delta = float(context.get("target_delta", 1e-8) or 1e-8)
    direction = centers - 0.5
    centers += direction * target_delta * 0.08
    eps = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], eps, 1.0 - eps)
    centers[:, 1] = np.clip(centers[:, 1], eps, 1.0 - eps)
    return {"centers": centers, "radii": radii,
            "metadata": {"strategy_family": "depth_refinement", "target_delta": target_delta}}'''


def _strategy_family(operator: str) -> str:
    return {
        "parameter_mutation": "parameter_mutation",
        "solver_switch": "solver_switch",
        "contact_threshold_mutation": "contact_threshold_mutation",
        "program_patch": "program_patch",
        "crossover": "crossover",
        "depth_refinement": "depth_refinement",
    }.get(operator, "self_evolve")


def _metadata_context(context: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {}
    for key, value in context.items():
        if key == "mate":
            metadata[key] = {"available": bool(value)}
        elif "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            metadata[key] = "<redacted>"
        else:
            metadata[key] = value
    return metadata


def _llm_fallback_reason(operator: str, context: Dict[str, Any]) -> Optional[str]:
    if operator != "program_patch":
        return None
    if not context.get("use_llm"):
        return "LLM program patch disabled for this run."
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY not present in environment; deterministic fallback used."
    return "Deterministic fallback retained to avoid network dependency in this harness run."
