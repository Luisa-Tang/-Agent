"""Logging helpers for machine-readable and human-readable agent traces."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from archive import sanitize_json


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def reset_human_log(path: Path, task: str, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Agent run log for Task {task}\n"
        f"run_id: {run_id}\n"
        f"started_utc: {utc_stamp()}\n\n",
        encoding="utf-8",
    )


def append_human_iteration(path: Path, record: Dict[str, Any], improved: bool) -> None:
    llm = record.get("llm_decision") or {}
    llm_line = "  llm_reflection: disabled"
    if llm.get("enabled"):
        status = "used" if llm.get("used") else "fallback"
        llm_line = (
            f"  llm_reflection: {status} | suggested: {llm.get('strategy')} | "
            f"reason: {llm.get('reason', '')}"
        )
    trace = record.get("trace") or {}
    observation = trace.get("observation") or {}
    thought = trace.get("thought_decision") or {}
    action = trace.get("action") or {}
    next_observation = trace.get("next_observation") or {}
    metrics = record.get("geometry_metrics") or {}
    lines = [
        f"Iteration {record['iteration']} | candidate {record['candidate_id']}",
        f"  observe: last={observation.get('last_evaluator_result')} | best={observation.get('best_archive_state')}",
        f"  think: selected={thought.get('selected_strategy', record['strategy'])} | reason={thought.get('reason', record.get('decision', ''))}",
        f"  act: {action.get('optimizer_or_repair', record['strategy'])} | code_bytes={action.get('generated_code_bytes', 0)}",
        f"  skills_used: {', '.join(record.get('skills_used') or action.get('skills_used') or [])}",
        f"  next_observe: valid={next_observation.get('valid', record['valid'])} | score={float(next_observation.get('score') or 0.0):.6f} | failure={next_observation.get('failure_type', record.get('failure_type'))}",
        f"  strategy: {record['strategy']}",
        f"  local_policy_strategy: {record.get('local_policy_strategy', record['strategy'])}",
        llm_line,
        f"  geometry: min_pairwise_margin={float(metrics.get('min_pairwise_margin') or 0.0):.3e} | "
        f"min_boundary_margin={float(metrics.get('min_boundary_margin') or 0.0):.3e}",
        f"  valid: {record['valid']} | score: {float(record.get('score') or 0.0):.6f} | "
        f"sum_radii: {float(record.get('sum_radii') or 0.0):.6f}",
        f"  failure_type: {record.get('failure_type', 'unknown')}",
        f"  archive_update: {'new best valid candidate' if improved else 'no best improvement'}",
        f"  decision: {record.get('decision', '')}",
        f"  raw_output: {record.get('raw_output', '')}",
        f"  code_snapshot: {record.get('code_snapshot', '')}",
        "",
    ]
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sanitize_json(record), sort_keys=True) + "\n")


def append_final_summary(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\nFinal summary\n")
        fh.write(text.rstrip() + "\n")
