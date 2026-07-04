"""Prompt artifact builder for GeoEvolve-lite.

The deterministic harness works without an LLM. When an LLM is unavailable,
these prompt files still provide replayable evidence for what a program-patch
operator would have asked a model to change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from archive import sanitize_json


class PromptSampler:
    def __init__(self, repo_root: Path, prompt_root: Path):
        self.repo_root = Path(repo_root)
        self.prompt_root = Path(prompt_root)
        self.template_root = Path(__file__).resolve().parent / "prompts"
        self.prompt_root.mkdir(parents=True, exist_ok=True)

    def build_prompt(self, task: str, operator: str, parent_record: Dict[str, Any],
                     elites: Iterable[Dict[str, Any]],
                     recent_failures: Iterable[Dict[str, Any]],
                     operator_stats: Dict[str, Any],
                     context: Dict[str, Any]) -> str:
        template_name = {
            "depth_refinement": "depth_refine.md",
            "crossover": "crossover.md",
            "block_crossover": "crossover_blocks.md",
            "boundary_slide_mutation": "mutate_geometry_block.md",
            "contact_pair_relaxation": "mutate_geometry_block.md",
            "small_circle_reposition": "mutate_geometry_block.md",
            "boundary_pattern_swap": "mutate_geometry_block.md",
            "radius_group_redistribution": "mutate_geometry_block.md",
            "aspect_ratio_sweep_local": "mutate_geometry_block.md",
            "contact_graph_breaking_refine": "mutate_geometry_block.md",
            "contact_graph_preserving_refine": "mutate_refiner_block.md",
            "solver_switch": "mutate_radius_solver_block.md",
        }.get(operator, "mutate_program.md")
        template = (self.template_root / template_name).read_text(encoding="utf-8")
        payload = {
            "task": task,
            "operator": operator,
            "parent_program": _compact_program(parent_record),
            "diverse_elites": [_compact_program(item) for item in elites],
            "recent_failures": [_compact_failure(item) for item in recent_failures],
            "operator_stats": operator_stats,
            "context": _redact_context(context),
            "target": "Improve score while preserving official evaluator validity.",
            "hard_rule": "Only modify code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END.",
        }
        return template + "\n\n```json\n" + json.dumps(sanitize_json(payload), indent=2, sort_keys=True) + "\n```\n"

    def save_prompt(self, program_id: str, prompt: str) -> Path:
        path = self.prompt_root / f"{program_id}.md"
        path.write_text(prompt, encoding="utf-8")
        return path


def _compact_program(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "program_id": record.get("program_id"),
        "parent_program_id": record.get("parent_program_id"),
        "task": record.get("task"),
        "operator": record.get("operator"),
        "score": record.get("score"),
        "sum_radii": record.get("sum_radii"),
        "valid": record.get("valid"),
        "contact_graph_hash": record.get("contact_graph_hash"),
        "boundary_pattern": record.get("boundary_pattern"),
        "novelty_score": record.get("novelty_score"),
        "strategy_family": record.get("strategy_family"),
        "metadata": _compact_metadata(record.get("metadata") or {}),
    }


def _compact_failure(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "program_id": record.get("program_id"),
        "task": record.get("task"),
        "operator": record.get("operator"),
        "failure_type": (record.get("metadata") or {}).get("failure_type"),
        "decision_reason": (record.get("metadata") or {}).get("decision_reason"),
        "score": record.get("score"),
        "novelty_score": record.get("novelty_score"),
    }


def _compact_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "failure_type",
        "decision_reason",
        "internal_valid",
        "internal_message",
        "strategy_family",
        "operator_context",
    ]
    return {key: metadata.get(key) for key in keep if key in metadata}


def _redact_context(context: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for key, value in context.items():
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            redacted[key] = "<redacted>"
        elif key == "mate" and isinstance(value, dict):
            redacted[key] = {
                "has_centers": "centers" in value,
                "has_radii": "radii" in value,
                "width": value.get("width"),
                "height": value.get("height"),
            }
        elif key == "mate_program_paths" and isinstance(value, dict):
            redacted[key] = dict(value)
        elif callable(value):
            redacted[key] = f"<callable:{getattr(value, '__name__', 'anonymous')}>"
        else:
            redacted[key] = value
    return redacted
