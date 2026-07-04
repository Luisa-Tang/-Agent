"""Program-level and block-level mutation operators for GeoEvolve-lite."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .program_db import ProgramDatabase, ProgramRecord, code_hash
except ImportError:  # pragma: no cover - direct module execution fallback
    from program_db import ProgramDatabase, ProgramRecord, code_hash


EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"
BLOCKS = {
    "aspect": ("# EVOLVE-BLOCK-ASPECT-START", "# EVOLVE-BLOCK-ASPECT-END"),
    "geometry": ("# EVOLVE-BLOCK-GEOMETRY-START", "# EVOLVE-BLOCK-GEOMETRY-END"),
    "radius": ("# EVOLVE-BLOCK-RADIUS-START", "# EVOLVE-BLOCK-RADIUS-END"),
    "refine": ("# EVOLVE-BLOCK-REFINE-START", "# EVOLVE-BLOCK-REFINE-END"),
    "safety": ("# EVOLVE-BLOCK-SAFETY-START", "# EVOLVE-BLOCK-SAFETY-END"),
}


V2_GEOMETRY_OPERATORS = {
    "boundary_slide_mutation",
    "contact_pair_relaxation",
    "small_circle_reposition",
    "boundary_pattern_swap",
    "radius_group_redistribution",
    "aspect_ratio_sweep_local",
    "contact_graph_breaking_refine",
}
V2_REFINE_OPERATORS = {"contact_graph_preserving_refine"}
V2_OPERATORS = V2_GEOMETRY_OPERATORS | V2_REFINE_OPERATORS | {"block_crossover"}
V3_RISKY_OPERATORS = {
    "destroy_repair_k_small",
    "gap_insertion_search",
    "small_circle_swap",
    "boundary_gap_refill",
    "contact_edge_break_then_repair",
    "aspect_ratio_island",
}
V3_OPERATORS = V2_OPERATORS | V3_RISKY_OPERATORS


def template_program_code(task: str) -> str:
    task = task.upper()
    name = "task_a_candidate_program.py" if task == "A" else "task_b_candidate_program.py"
    return (Path(__file__).resolve().parent / "evolve_blocks" / name).read_text(encoding="utf-8")


def create_seed_program(db: ProgramDatabase, task: str, data_metadata: Dict[str, Any]) -> ProgramRecord:
    program_id = db.next_program_id(task)
    code = template_program_code(task)
    path = db.write_program(program_id, code)
    hashes = block_hashes(code)
    return db.add(
        ProgramRecord(
            program_id=program_id,
            parent_program_id=None,
            task=task.upper(),
            operator="seed_parent_program",
            operator_name="seed_parent_program",
            code_path=str(path),
            code_hash=code_hash(code),
            block_hashes=hashes,
            blocks_used=[],
            block_types_changed=[],
            score=float(data_metadata.get("score") or 0.0),
            sum_radii=float(data_metadata.get("sum_radii") or 0.0),
            valid=True,
            official_valid=True,
            official_score=float(data_metadata.get("score") or 0.0),
            contact_graph_hash=data_metadata.get("contact_graph_hash"),
            boundary_pattern=data_metadata.get("boundary_pattern"),
            novelty_score=1.0,
            strategy_family="seed_parent_program",
            cascade_stage_reached="seed",
            metadata=dict(data_metadata),
        )
    )


def create_child_program(db: ProgramDatabase, task: str, parent_record: ProgramRecord,
                         operator: str, rng: np.random.Generator,
                         context: Dict[str, Any],
                         prompt_path: Optional[Path] = None) -> ProgramRecord:
    parent_code = Path(parent_record.code_path).read_text(encoding="utf-8")
    if context.get("evolve_blocks_v2"):
        child_code, changed_blocks = mutate_v2_blocks(parent_code, task, operator, context)
    else:
        block = _operator_block(task, operator, rng, context)
        child_code = replace_evolve_block(parent_code, block)
        changed_blocks = ["outer"]
    program_id = db.next_program_id(task)
    child_path = db.write_program(program_id, child_code)
    hashes = block_hashes(child_code)
    metadata = {
        "operator_context": _metadata_context(context),
        "prompt_path": str(prompt_path) if prompt_path else None,
        "parent_code_hash": parent_record.code_hash,
        "parent_block_hashes": parent_record.block_hashes,
        "block_hashes": hashes,
        "block_types_changed": changed_blocks,
        "llm_used": False,
        "llm_fallback_reason": _llm_fallback_reason(operator, context),
        "inspirations": ["OpenEvolve", "ShinkaEvolve", "CodeEvolve"],
        "note": "EVOLVE-BLOCK mutation; final solution export still requires official evaluator validity.",
    }
    return db.add(
        ProgramRecord(
            program_id=program_id,
            parent_program_id=parent_record.program_id,
            task=task.upper(),
            operator=operator,
            operator_name=operator,
            code_path=str(child_path),
            code_hash=code_hash(child_code),
            block_hashes=hashes,
            blocks_used=[],
            block_types_changed=changed_blocks,
            strategy_family=_strategy_family(operator),
            metadata=metadata,
        )
    )


def mutate_v2_blocks(parent_code: str, task: str, operator: str,
                     context: Dict[str, Any]) -> Tuple[str, List[str]]:
    task = task.upper()
    if not all(marker in parent_code for pair in BLOCKS.values() for marker in pair if not (task == "B" and "ASPECT" in marker)):
        parent_code = template_program_code(task)
    if operator == "block_crossover":
        return _block_crossover(parent_code, context)
    if operator == "aspect_ratio_sweep_local" and task == "A":
        return replace_named_block(parent_code, "aspect", _aspect_ratio_sweep_block()), ["aspect"]
    if operator == "aspect_ratio_island" and task == "A":
        code = replace_named_block(parent_code, "aspect", _aspect_ratio_island_block())
        code = replace_named_block(code, "geometry", _gap_refill_block(task, operator))
        return code, ["aspect", "geometry"]
    if operator in {"destroy_repair_k_small", "gap_insertion_search", "small_circle_swap", "boundary_gap_refill"}:
        return replace_named_block(parent_code, "geometry", _gap_refill_block(task, operator)), ["geometry"]
    if operator == "contact_edge_break_then_repair":
        return replace_named_block(parent_code, "geometry", _contact_edge_break_then_repair_block(task)), ["geometry"]
    if operator in V2_GEOMETRY_OPERATORS:
        return replace_named_block(parent_code, "geometry", _geometry_v2_block(task, operator)), ["geometry"]
    if operator in V2_REFINE_OPERATORS:
        return replace_named_block(parent_code, "refine", _refine_v2_block(task, operator)), ["refine"]
    if operator == "solver_switch":
        return replace_named_block(parent_code, "radius", _radius_solver_v2_block()), ["radius"]
    if operator == "program_patch":
        return replace_named_block(parent_code, "geometry", _geometry_v2_block(task, "small_circle_reposition")), ["geometry"]
    block = _operator_block(task, operator, np.random.default_rng(0), context)
    return replace_evolve_block(parent_code, block), ["outer"]


def _block_crossover(parent_code: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
    code = parent_code
    changed = []
    mate_paths = context.get("mate_program_paths") or {}
    for block_name in ("geometry", "radius", "refine"):
        path_value = mate_paths.get(block_name)
        if not path_value:
            continue
        path = Path(str(path_value))
        if not path.exists():
            continue
        try:
            mate_code = path.read_text(encoding="utf-8")
            block = extract_named_block(mate_code, block_name)
        except Exception:
            continue
        code = replace_named_block(code, block_name, block)
        changed.append(block_name)
    if not changed:
        code = replace_named_block(code, "geometry", _geometry_v2_block(str(context.get("task") or "B"), "boundary_pattern_swap"))
        changed.append("geometry")
    return code, changed


def extract_evolve_block(code: str) -> str:
    start = code.index(EVOLVE_START)
    end = code.index(EVOLVE_END, start)
    return code[start + len(EVOLVE_START):end].strip()


def replace_evolve_block(code: str, block: str) -> str:
    start = code.index(EVOLVE_START)
    end = code.index(EVOLVE_END, start) + len(EVOLVE_END)
    replacement = EVOLVE_START + "\n" + block.rstrip() + "\n" + EVOLVE_END
    return code[:start] + replacement + code[end:]


def extract_named_block(code: str, block_name: str) -> str:
    start_marker, end_marker = BLOCKS[block_name]
    start = code.index(start_marker)
    end = code.index(end_marker, start)
    return code[start + len(start_marker):end].strip("\n")


def replace_named_block(code: str, block_name: str, block: str) -> str:
    start_marker, end_marker = BLOCKS[block_name]
    start = code.index(start_marker)
    end = code.index(end_marker, start) + len(end_marker)
    replacement = start_marker + "\n" + block.rstrip() + "\n    " + end_marker
    return code[:start] + replacement + code[end:]


def block_hashes(code: str) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for name in BLOCKS:
        try:
            block = extract_named_block(code, name)
        except Exception:
            continue
        hashes[name] = hashlib.sha256(block.encode("utf-8")).hexdigest()
    return hashes


def _geometry_v2_block(task: str, operator: str) -> str:
    task = task.upper()
    if operator == "boundary_slide_mutation":
        return _boundary_slide_block(task)
    if operator == "contact_pair_relaxation":
        return _contact_pair_relaxation_block(task)
    if operator == "small_circle_reposition":
        return _small_circle_reposition_block(task)
    if operator == "boundary_pattern_swap":
        return _boundary_pattern_swap_block(task)
    if operator == "radius_group_redistribution":
        return _radius_group_redistribution_block(task)
    if operator == "aspect_ratio_sweep_local":
        return _aspect_geometry_rescale_block()
    if operator == "contact_graph_breaking_refine":
        return _contact_graph_breaking_block(task)
    return _boundary_slide_block(task)


def _boundary_slide_block(task: str) -> str:
    if task == "A":
        edge_expr = "np.minimum.reduce([centers[:, 0], width - centers[:, 0], centers[:, 1], height - centers[:, 1]])"
        clip_x = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)"
        clip_y = "centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)"
    else:
        edge_expr = "np.minimum.reduce([centers[:, 0], 1.0 - centers[:, 0], centers[:, 1], 1.0 - centers[:, 1]])"
        clip_x = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)"
        clip_y = "centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)"
    return f'''    edge_margin = {edge_expr}
    band = float(context.get("edge_band", 5e-7) or 5e-7)
    step = float(context.get("sigma", 3e-7) or 3e-7)
    idx = np.where(edge_margin <= np.percentile(edge_margin, 35))[0]
    changed_indices = idx.astype(int).tolist()
    for k in idx:
        left = centers[k, 0]
        right = (width - centers[k, 0]) if metadata.get("task") == "A" else (1.0 - centers[k, 0])
        bottom = centers[k, 1]
        top = (height - centers[k, 1]) if metadata.get("task") == "A" else (1.0 - centers[k, 1])
        side = int(np.argmin([left, right, bottom, top]))
        if side in (0, 1):
            centers[k, 1] += rng.normal(0.0, step)
        else:
            centers[k, 0] += rng.normal(0.0, step)
    {clip_x}
    {clip_y}
    metadata["blocks_used"].append("geometry:boundary_slide")
    metadata["strategy_family"] = "boundary_slide_mutation"
    metadata["operator_name"] = "boundary_slide_mutation"
    metadata["intended_contact_change"] = "preserve local boundary contacts while changing tangential order"
    metadata["intended_boundary_change"] = "slide active boundary circles"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {{"edge_band": band, "step": step}}'''


def _contact_pair_relaxation_block(task: str) -> str:
    if task == "A":
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)"
    else:
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)"
    return f'''    n = len(radii)
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                pairs.append((d - float(radii[i] + radii[j]), i, j))
        pairs.sort(key=lambda item: item[0])
        _m, i, j = pairs[int(context.get("pair_rank", 0) or 0) % len(pairs)]
        direction = centers[i] - centers[j]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction = direction / norm
            eps = float(context.get("target_delta", 1e-8) or 1e-8)
            centers[i] += eps * direction
            centers[j] -= eps * direction
            changed_indices = [int(i), int(j)]
        else:
            changed_indices = []
    else:
        changed_indices = []
    {clip}
    metadata["blocks_used"].append("geometry:contact_pair_relaxation")
    metadata["strategy_family"] = "contact_pair_relaxation"
    metadata["operator_name"] = "contact_pair_relaxation"
    metadata["intended_contact_change"] = "relax one tight circle pair"
    metadata["intended_boundary_change"] = "none"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {{"target_delta": float(context.get("target_delta", 1e-8) or 1e-8)}}'''


def _small_circle_reposition_block(task: str) -> str:
    if task == "A":
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)"
        size = "np.array([width, height], dtype=float)"
    else:
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)"
        size = "np.array([1.0, 1.0], dtype=float)"
    return f'''    count = max(1, int(context.get("small_circle_count", 2) or 2))
    order = np.argsort(radii)[:count]
    box = {size}
    changed_indices = []
    anchors = centers[np.argsort(-radii)[: max(1, min(5, len(radii)))]]
    for rank, idx in enumerate(order):
        if len(anchors):
            anchor = anchors[int(rng.integers(0, len(anchors)))]
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            dist = float(context.get("gap_probe_scale", 0.035) or 0.035) * min(box)
            centers[idx] = anchor + dist * np.array([np.cos(angle), np.sin(angle)])
        else:
            centers[idx] = rng.uniform(0.05, 0.95, size=2) * box
        changed_indices.append(int(idx))
    {clip}
    metadata["blocks_used"].append("geometry:small_circle_reposition")
    metadata["strategy_family"] = "small_circle_reposition"
    metadata["operator_name"] = "small_circle_reposition"
    metadata["intended_contact_change"] = "move smallest circles into nearby gaps"
    metadata["intended_boundary_change"] = "possible"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {{"small_circle_count": count}}'''


def _boundary_pattern_swap_block(task: str) -> str:
    if task == "A":
        limit = "np.array([width, height], dtype=float)"
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)"
    else:
        limit = "np.array([1.0, 1.0], dtype=float)"
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)"
    return f'''    box = {limit}
    margins = np.minimum.reduce([centers[:, 0], box[0] - centers[:, 0], centers[:, 1], box[1] - centers[:, 1]])
    idx = np.argsort(margins)[: max(2, min(6, len(margins)))]
    changed_indices = []
    if len(idx) >= 2:
        a = int(idx[int(rng.integers(0, len(idx)))])
        b = int(idx[int(rng.integers(0, len(idx)))])
        if a != b:
            if rng.random() < 0.5:
                centers[a, 0], centers[b, 0] = centers[b, 0], centers[a, 0]
            else:
                centers[a, 1], centers[b, 1] = centers[b, 1], centers[a, 1]
            changed_indices = [a, b]
    {clip}
    metadata["blocks_used"].append("geometry:boundary_pattern_swap")
    metadata["strategy_family"] = "boundary_pattern_swap"
    metadata["operator_name"] = "boundary_pattern_swap"
    metadata["intended_contact_change"] = "change boundary-neighbor assignments"
    metadata["intended_boundary_change"] = "swap active boundary positions"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {{"candidate_count": int(len(idx))}}'''


def _radius_group_redistribution_block(task: str) -> str:
    if task == "A":
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)"
        mid = "np.array([0.5 * width, 0.5 * height])"
    else:
        clip = "centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)\n    centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)"
        mid = "np.array([0.5, 0.5])"
    return f'''    q1, q2 = np.quantile(radii, [0.33, 0.67])
    small = np.where(radii <= q1)[0]
    medium = np.where((radii > q1) & (radii <= q2))[0]
    center = {mid}
    step = float(context.get("sigma", 3e-7) or 3e-7)
    changed_indices = []
    for idx in small:
        centers[idx] += rng.normal(0.0, 2.5 * step, size=2)
        changed_indices.append(int(idx))
    for idx in medium:
        direction = centers[idx] - center
        norm = max(float(np.linalg.norm(direction)), 1e-12)
        centers[idx] += direction / norm * rng.normal(0.0, step)
        changed_indices.append(int(idx))
    {clip}
    metadata["blocks_used"].append("geometry:radius_group_redistribution")
    metadata["strategy_family"] = "radius_group_redistribution"
    metadata["operator_name"] = "radius_group_redistribution"
    metadata["intended_contact_change"] = "small circles explore, large circles remain stable"
    metadata["intended_boundary_change"] = "possible"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {{"q1": float(q1), "q2": float(q2), "step": step}}'''


def _aspect_ratio_sweep_block() -> str:
    return '''    old_width = float(width)
    old_height = float(height)
    delta = float(context.get("aspect_delta", context.get("target_delta", 1e-7)) or 1e-7)
    direction = float(context.get("width_direction", 1.0) or 1.0)
    width = float(np.clip(width + direction * delta, 0.28, 1.72))
    height = 2.0 - width
    x_scale = width / max(old_width, 1e-12)
    y_scale = height / max(old_height, 1e-12)
    centers[:, 0] *= x_scale
    centers[:, 1] *= y_scale
    metadata["blocks_used"].append("aspect:aspect_ratio_sweep_local")
    metadata["operator_name"] = "aspect_ratio_sweep_local"
    metadata["parameters"] = {"aspect_delta": delta, "width_direction": direction}'''


def _aspect_ratio_island_block() -> str:
    return '''    old_width = float(width)
    old_height = float(height)
    deltas = context.get("aspect_bucket_deltas") or [1e-5, 3e-5, 1e-4, 3e-4]
    delta = float(deltas[int(context.get("generation", 0) or 0) % len(deltas)])
    direction = float(context.get("width_direction", 1.0) or 1.0)
    width = float(np.clip(width + direction * delta, 0.28, 1.72))
    height = 2.0 - width
    centers[:, 0] *= width / max(old_width, 1e-12)
    centers[:, 1] *= height / max(old_height, 1e-12)
    old_bucket = int(np.floor(old_width / 3e-5))
    new_bucket = int(np.floor(width / 3e-5))
    metadata["blocks_used"].append("aspect:aspect_ratio_island")
    metadata["operator_name"] = "aspect_ratio_island"
    metadata["old_width"] = old_width
    metadata["new_width"] = width
    metadata["width_delta"] = width - old_width
    metadata["aspect_bucket"] = new_bucket
    metadata["aspect_ratio_bucket_changed"] = bool(old_bucket != new_bucket)
    metadata["parameters"] = {"aspect_delta": delta, "width_direction": direction}'''


def _gap_refill_block(task: str, operator: str) -> str:
    task = task.upper()
    if task == "A":
        call = "gap_helper(centers=centers, radii=radii, width=width, height=height, k=k, mode=mode)"
    else:
        call = "gap_helper(centers=centers, radii=radii, width=1.0, height=1.0, k=k, mode=mode)"
    default_k = "3" if operator in {"destroy_repair_k_small", "gap_insertion_search"} else "2"
    return f'''    mode = "{operator}"
    k = int(context.get("destroy_k", {default_k}) or {default_k})
    if mode == "small_circle_swap":
        k = max(2, min(k, 3))
    elif mode == "boundary_gap_refill":
        k = max(1, min(k, 2))
    else:
        k = max(2, min(k, 4))
    gap_helper = context.get("gap_refill")
    if callable(gap_helper):
        centers, gap_meta = {call}
    else:
        order = np.argsort(radii)[:k]
        gap_meta = {{"removed_indices": [int(i) for i in order], "gap_ids": [], "small_circle_reassigned": True}}
        for idx in order:
            centers[idx] += rng.normal(0.0, float(context.get("gap_probe_scale", 0.035) or 0.035), size=2)
    changed_indices = [int(i) for i in gap_meta.get("removed_indices", [])]
    metadata["blocks_used"].append("geometry:{operator}")
    metadata["strategy_family"] = "{operator}"
    metadata["operator_name"] = "{operator}"
    metadata["intended_contact_change"] = "destroy small-circle placement and refill high-scoring gaps"
    metadata["intended_boundary_change"] = "possible boundary gap refill"
    metadata["changed_indices"] = changed_indices
    metadata["removed_indices"] = changed_indices
    metadata["inserted_gap_ids"] = list(gap_meta.get("gap_ids", []))
    metadata["gap_sources"] = list(gap_meta.get("gap_sources", []))
    metadata["small_circle_reassigned"] = bool(gap_meta.get("small_circle_reassigned", False))
    metadata["top_gap_score"] = float(gap_meta.get("top_gap_score", 0.0) or 0.0)
    metadata["top_gap_radius"] = float(gap_meta.get("top_gap_radius", 0.0) or 0.0)
    metadata["parameters"] = {{"destroy_k": k, "mode": mode}}'''


def _contact_edge_break_then_repair_block(task: str) -> str:
    if task.upper() == "A":
        clip_comment = "# risky island intentionally allows pre-repair out-of-bounds before cascade repair"
    else:
        clip_comment = "# risky island intentionally allows pre-repair out-of-bounds before cascade repair"
    return f'''    n = len(radii)
    changed_indices = []
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                local_degree = 1.0
                stress = float(radii[i] + radii[j]) / max(d, 1e-12) / local_degree
                pairs.append((abs(d - float(radii[i] + radii[j])), stress, i, j))
        pairs.sort(key=lambda item: (item[0], item[1]))
        _m, _stress, i, j = pairs[int(context.get("pair_rank", 0) or 0) % len(pairs)]
        direction = centers[i] - centers[j]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction = direction / norm
            eps = float(context.get("edge_break_eps", context.get("sigma", 1e-5)) or 1e-5)
            centers[i] += eps * direction
            centers[j] -= eps * direction
            changed_indices = [int(i), int(j)]
    {clip_comment}
    metadata["blocks_used"].append("geometry:contact_edge_break_then_repair")
    metadata["strategy_family"] = "contact_edge_break_then_repair"
    metadata["operator_name"] = "contact_edge_break_then_repair"
    metadata["intended_contact_change"] = "break one low-value tight edge before LP repair"
    metadata["intended_boundary_change"] = "possible"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {{"edge_break_eps": float(context.get("edge_break_eps", context.get("sigma", 1e-5)) or 1e-5)}}'''


def _aspect_geometry_rescale_block() -> str:
    return '''    centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)
    centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)
    metadata["blocks_used"].append("geometry:aspect_rescale_clip")
    metadata["strategy_family"] = "aspect_ratio_sweep_local"
    metadata["intended_contact_change"] = "aspect ratio rescale changes pair distances"
    metadata["intended_boundary_change"] = "preserve scaled boundary pattern"
    metadata["changed_indices"] = list(range(len(centers)))'''


def _contact_graph_breaking_block(task: str) -> str:
    return _contact_pair_relaxation_block(task).replace(
        "contact_pair_relaxation",
        "contact_graph_breaking_refine",
    ).replace(
        "relax one tight circle pair",
        "break one active edge to search a new basin",
    )


def _refine_v2_block(task: str, operator: str) -> str:
    if task.upper() == "A":
        return '''    max_steps = int(context.get("max_refine_steps", 4) or 4)
    changed_indices = []
    for _ in range(max_steps):
        centers[:, 0] = np.clip(centers[:, 0], 1e-9, width - 1e-9)
        centers[:, 1] = np.clip(centers[:, 1], 1e-9, height - 1e-9)
    metadata["blocks_used"].append("refine:contact_graph_preserving_refine")
    metadata["strategy_family"] = "contact_graph_preserving_refine"
    metadata["operator_name"] = "contact_graph_preserving_refine"
    metadata["intended_contact_change"] = "preserve active graph while LP reallocates radii"
    metadata["intended_boundary_change"] = "preserve"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"max_steps": max_steps}'''
    return '''    max_steps = int(context.get("max_refine_steps", 4) or 4)
    changed_indices = []
    for _ in range(max_steps):
        centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1.0 - 1e-9)
        centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1.0 - 1e-9)
    metadata["blocks_used"].append("refine:contact_graph_preserving_refine")
    metadata["strategy_family"] = "contact_graph_preserving_refine"
    metadata["operator_name"] = "contact_graph_preserving_refine"
    metadata["intended_contact_change"] = "preserve active graph while LP reallocates radii"
    metadata["intended_boundary_change"] = "preserve"
    metadata["changed_indices"] = changed_indices
    metadata["parameters"] = {"max_steps": max_steps}'''


def _radius_solver_v2_block() -> str:
    return '''    solver = context.get("solve_radius_lp")
    if callable(solver):
        radii = solver(centers=centers, container=(width, height) if metadata.get("task") == "A" else (1.0, 1.0), task=metadata.get("task"))
    else:
        radii = np.maximum(radii, 0.0)
    metadata["blocks_used"].append("radius:solver_switch_lp")
    metadata["operator_name"] = "solver_switch"
    metadata["parameters"] = {"solver": context.get("solver", "lp")}'''


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
        return _parameter_mutation_block(task).replace("parameter_mutation", "solver_switch")
    return _parameter_mutation_block(task).replace("parameter_mutation", "solver_switch")


def _contact_threshold_block(task: str) -> str:
    if task == "A":
        return _parameter_mutation_block(task).replace("parameter_mutation", "contact_threshold_mutation")
    return _parameter_mutation_block(task).replace("parameter_mutation", "contact_threshold_mutation")


def _program_patch_fallback_block(task: str) -> str:
    if task == "A":
        return _parameter_mutation_block(task).replace("parameter_mutation", "program_patch_fallback")
    return _parameter_mutation_block(task).replace("parameter_mutation", "program_patch_fallback")


def _crossover_block(task: str) -> str:
    if task == "A":
        return _parameter_mutation_block(task).replace("parameter_mutation", "crossover")
    return _parameter_mutation_block(task).replace("parameter_mutation", "crossover")


def _depth_refinement_block(task: str) -> str:
    if task == "A":
        return _parameter_mutation_block(task).replace("parameter_mutation", "depth_refinement")
    return _parameter_mutation_block(task).replace("parameter_mutation", "depth_refinement")


def _strategy_family(operator: str) -> str:
    mapping = {
        "parameter_mutation": "parameter_mutation",
        "solver_switch": "solver_switch",
        "contact_threshold_mutation": "contact_threshold_mutation",
        "program_patch": "program_patch",
        "crossover": "crossover",
        "depth_refinement": "depth_refinement",
        "boundary_slide_mutation": "boundary_slide_mutation",
        "contact_pair_relaxation": "contact_pair_relaxation",
        "small_circle_reposition": "small_circle_reposition",
        "boundary_pattern_swap": "boundary_pattern_swap",
        "radius_group_redistribution": "radius_group_redistribution",
        "aspect_ratio_sweep_local": "aspect_ratio_sweep_local",
        "contact_graph_preserving_refine": "contact_graph_preserving_refine",
        "contact_graph_breaking_refine": "contact_graph_breaking_refine",
        "block_crossover": "block_crossover",
        "destroy_repair_k_small": "destroy_repair_k_small",
        "gap_insertion_search": "gap_insertion_search",
        "small_circle_swap": "small_circle_swap",
        "boundary_gap_refill": "boundary_gap_refill",
        "contact_edge_break_then_repair": "contact_edge_break_then_repair",
        "aspect_ratio_island": "aspect_ratio_island",
    }
    return mapping.get(operator, "self_evolve")


def _metadata_context(context: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {}
    for key, value in context.items():
        if key == "mate":
            metadata[key] = {"available": bool(value)}
        elif key == "mate_program_paths":
            metadata[key] = dict(value) if isinstance(value, dict) else {}
        elif callable(value):
            metadata[key] = f"<callable:{getattr(value, '__name__', 'anonymous')}>"
        elif "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            metadata[key] = "<redacted>"
        else:
            metadata[key] = value
    return metadata


def _llm_fallback_reason(operator: str, context: Dict[str, Any]) -> Optional[str]:
    if operator not in {"program_patch", "boundary_slide_mutation", "small_circle_reposition", "block_crossover"}:
        return None
    if not context.get("use_llm"):
        return "LLM block patch disabled for this run."
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY not present in environment; deterministic fallback used."
    return "Deterministic fallback retained to avoid network dependency in this harness run."
