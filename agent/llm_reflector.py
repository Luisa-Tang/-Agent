"""Optional LLM-backed strategy reflection for the local Agent.

The Agent remains deterministic by default. This module is only used when the
operator passes ``--use-llm`` and provides credentials through environment
variables; failures return a local fallback suggestion instead of stopping the
optimization loop.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ALLOWED_STRATEGIES = {
    "baseline_safe_grid",
    "scipy_slsqp_joint",
    "multi_start_slsqp",
    "hexagonal_or_staggered_initialization",
    "perturb_best_and_repair",
}


@dataclass
class LLMDecision:
    strategy: Optional[str]
    reason: str
    enabled: bool
    used: bool
    error: Optional[str] = None
    raw_excerpt: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "enabled": self.enabled,
            "used": self.used,
            "error": self.error,
            "raw_excerpt": self.raw_excerpt,
        }


class LLMReflector:
    def __init__(self, enabled: bool, repo_root: Path, model: str,
                 base_url: Optional[str], timeout: float = 20.0):
        self.enabled = bool(enabled)
        self.repo_root = Path(repo_root)
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.api_key_env = "OPENAI_API_KEY"
        self._client = None

    def decide(self, task: str, iteration: int, best_record: Optional[Dict[str, Any]],
               last_record: Optional[Dict[str, Any]], no_improve: int,
               local_strategy: str, recent_records: Iterable[Dict[str, Any]]) -> LLMDecision:
        if not self.enabled:
            return LLMDecision(local_strategy, "LLM disabled; using deterministic local policy.", False, False)
        if not os.environ.get(self.api_key_env):
            return LLMDecision(
                local_strategy,
                "OPENAI_API_KEY is not set; using deterministic local policy.",
                True,
                False,
                error="missing_api_key",
            )

        try:
            client = self._get_client()
            messages = self._messages(task, iteration, best_record, last_record,
                                      no_improve, local_strategy, recent_records)
            content = self._complete(client, messages)
            parsed = self._parse_response(content)
            strategy = parsed.get("strategy") or local_strategy
            reason = parsed.get("reason") or "LLM returned a strategy suggestion."
            if strategy not in ALLOWED_STRATEGIES:
                return LLMDecision(
                    local_strategy,
                    f"LLM suggested unsupported strategy {strategy!r}; using local policy.",
                    True,
                    False,
                    error="unsupported_strategy",
                    raw_excerpt=content[:500],
                )
            return LLMDecision(strategy, reason, True, True, raw_excerpt=content[:500])
        except Exception as exc:  # LLM use must never break deterministic fallback.
            return LLMDecision(
                local_strategy,
                "LLM call failed; using deterministic local policy.",
                True,
                False,
                error=type(exc).__name__,
            )

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            self._client = "stdlib_http"
            return self._client

        kwargs = {"api_key": os.environ[self.api_key_env]}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _complete(self, client: Any, messages: List[Dict[str, str]]) -> str:
        if client == "stdlib_http":
            return self._complete_with_urllib(messages)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""

    def _complete_with_urllib(self, messages: List[Dict[str, str]]) -> str:
        if not self.base_url:
            raise RuntimeError("base_url is required when the openai SDK is not installed")
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ[self.api_key_env]}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
        return str(payload["choices"][0]["message"].get("content") or "")

    def _messages(self, task: str, iteration: int, best_record: Optional[Dict[str, Any]],
                  last_record: Optional[Dict[str, Any]], no_improve: int,
                  local_strategy: str, recent_records: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
        system = _read_prompt(self.repo_root / "agent" / "prompts" / "system_prompt.md")
        task_prompt = _read_prompt(
            self.repo_root / "agent" / "prompts" / ("task_a_prompt.md" if task == "A" else "task_b_prompt.md")
        )
        payload = {
            "task": task,
            "iteration": iteration,
            "allowed_strategies": sorted(ALLOWED_STRATEGIES),
            "local_policy_strategy": local_strategy,
            "no_improve_count": no_improve,
            "best_record": _compact_record(best_record),
            "last_record": _compact_record(last_record),
            "recent_records": [_compact_record(r) for r in list(recent_records)[-6:]],
            "instruction": (
                "Return strict JSON only with keys strategy and reason. The strategy "
                "must be exactly one allowed strategy. Do not include code."
            ),
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task_prompt + "\n\n" + json.dumps(payload, sort_keys=True)},
        ]

    @staticmethod
    def _parse_response(content: str) -> Dict[str, str]:
        content = content.strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                return {}
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            return {}
        return {
            "strategy": str(data.get("strategy", "")).strip(),
            "reason": str(data.get("reason", "")).strip(),
        }


def _read_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "You are a cautious local optimization Agent."


def _compact_record(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    keys = [
        "task",
        "iteration",
        "candidate_id",
        "strategy",
        "valid",
        "score",
        "sum_radii",
        "width",
        "height",
        "failure_type",
        "decision",
    ]
    compact = {k: record.get(k) for k in keys if k in record}
    diagnostics = record.get("diagnostics") or {}
    for key in ("boundary_margin", "overlap_margin", "internal_valid", "internal_message"):
        if key in diagnostics:
            compact[f"diagnostics_{key}"] = diagnostics[key]
    return compact
