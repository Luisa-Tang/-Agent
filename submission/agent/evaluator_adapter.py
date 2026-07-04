"""Adapter around the official evaluation scripts.

The adapter intentionally treats the official evaluators as the source of truth.
It writes candidate code to task_X/solution.py, invokes the evaluator through a
subprocess, captures raw output, and parses the public score lines.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EvalResult:
    task: str
    valid: bool
    score: float
    sum_radii: Optional[float]
    failure_type: str
    stdout: str
    stderr: str
    returncode: int
    elapsed_text: Optional[str] = None

    @property
    def raw_output(self) -> str:
        if self.stderr:
            return self.stdout + "\n[stderr]\n" + self.stderr
        return self.stdout


class EvaluatorAdapter:
    def __init__(self, repo_root: Path, python_executable: Optional[str] = None,
                 timeout: int = 1100):
        self.repo_root = Path(repo_root).resolve()
        self.python = python_executable or sys.executable
        self.timeout = int(timeout)

    def task_dir(self, task: str) -> Path:
        task = task.upper()
        return self.repo_root / ("task_A" if task == "A" else "task_B")

    def solution_path(self, task: str) -> Path:
        return self.task_dir(task) / "solution.py"

    def evaluator_path(self, task: str) -> Path:
        return self.task_dir(task) / "evaluate.py"

    def write_candidate(self, task: str, code: str) -> Path:
        path = self.solution_path(task)
        path.write_text(code, encoding="utf-8")
        return path

    def evaluate_task(self, task: str, candidate_code: Optional[str] = None) -> EvalResult:
        task = task.upper()
        if candidate_code is not None:
            solution = self.write_candidate(task, candidate_code)
        else:
            solution = self.solution_path(task)

        cmd = [self.python, str(self.evaluator_path(task)), str(solution)]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                env=env,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nTimeout after {self.timeout}s"
            return EvalResult(
                task=task,
                valid=False,
                score=0.0,
                sum_radii=None,
                failure_type="timeout",
                stdout=stdout,
                stderr=stderr,
                returncode=124,
            )

        return self.parse_result(task, stdout, stderr, returncode)

    def parse_result(self, task: str, stdout: str, stderr: str,
                     returncode: int) -> EvalResult:
        text = stdout + "\n" + stderr
        score = _last_float_for_label(text, "Score")
        sum_radii = _last_float_for_label(text, "sum_radii")
        elapsed = _last_text_for_label(text, "Elapsed")
        failure_type = classify_failure(text, returncode)
        valid = returncode == 0 and failure_type == "none" and score is not None and score > 0.0
        return EvalResult(
            task=task,
            valid=valid,
            score=float(score or 0.0),
            sum_radii=sum_radii,
            failure_type=failure_type,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            elapsed_text=elapsed,
        )

    def run_evaluate_all(self, filename: Optional[str] = "solution.py") -> subprocess.CompletedProcess:
        cmd = [self.python, str(self.repo_root / "evaluate_all.py")]
        if filename:
            cmd.append(filename)
        return subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout * 2,
        )


def _last_float_for_label(text: str, label: str) -> Optional[float]:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _last_text_for_label(text: str, label: str) -> Optional[str]:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*([^\n]+)")
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def classify_failure(text: str, returncode: int) -> str:
    lower = text.lower()
    if "overlap" in lower:
        return "overlap"
    if "outside" in lower or "extends outside" in lower:
        return "boundary_violation"
    if "non-finite" in lower or "nonfinite" in lower or "nan" in lower or "inf" in lower:
        return "nonfinite"
    if "shape" in lower:
        return "shape_error"
    if "perimeter" in lower or "expected 4.0" in lower:
        return "perimeter_error"
    if "negative radius" in lower:
        return "negative_radius"
    if "timeout" in lower:
        return "timeout"
    if "[failed]" in lower or returncode != 0:
        return "unknown"
    if "[invalid]" in lower:
        return "unknown"
    return "none"
