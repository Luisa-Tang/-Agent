"""Safety checks for protected files, final solutions, and archived logs."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROTECTED_RELATIVE_FILES = [
    "evaluate_all.py",
    "task_A/evaluate.py",
    "task_A/task_description.md",
    "task_B/evaluate.py",
    "task_B/task_description.md",
]
API_KEY_RE = re.compile(r"\b(?:sk|sk-ant|sk-proj)-[A-Za-z0-9_-]{10,}\b")
NETWORK_RE = re.compile(r"\b(requests|urllib|httpx|socket|subprocess|curl|wget|urlopen|connect)\b")
ALLOWED_SOLUTION_IMPORTS = {"numpy"}


class SafetyGuard:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self.pre_hashes: Dict[str, Optional[str]] = {}

    def capture_pre_run(self) -> Dict[str, Optional[str]]:
        self.pre_hashes = {rel: _sha256(self.repo_root / rel) for rel in PROTECTED_RELATIVE_FILES}
        return dict(self.pre_hashes)

    def check_post_run(self, run_id: str, write_submission: bool = True) -> Dict[str, Any]:
        post_hashes = {rel: _sha256(self.repo_root / rel) for rel in PROTECTED_RELATIVE_FILES}
        protected_hash_unchanged = {
            rel: self.pre_hashes.get(rel) == post_hashes.get(rel)
            for rel in PROTECTED_RELATIVE_FILES
        }
        protected_git_diff = _git_diff_protected(self.repo_root)
        final_solution_checks = {
            "task_A/solution.py": self._check_solution(self.repo_root / "task_A" / "solution.py"),
            "task_B/solution.py": self._check_solution(self.repo_root / "task_B" / "solution.py"),
        }
        secret_scan = self._scan_api_keys(run_id)
        report = {
            "run_id": run_id,
            "protected_files": {
                "pre_hashes": self.pre_hashes,
                "post_hashes": post_hashes,
                "hash_unchanged": protected_hash_unchanged,
                "git_diff_protected_files": protected_git_diff,
                "unchanged": all(protected_hash_unchanged.values()) and not protected_git_diff,
            },
            "final_solutions": final_solution_checks,
            "secret_scan": secret_scan,
        }
        report["passed"] = (
            report["protected_files"]["unchanged"]
            and all(item["passed"] for item in final_solution_checks.values())
            and not secret_scan["matches"]
        )
        metrics_dir = self.repo_root / "agent" / "archive" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "safety_report.json"
        metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if write_submission:
            submission_path = self.repo_root / "submission" / "safety_report.json"
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            submission_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    def _check_solution(self, path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        imports = _solution_imports(text)
        disallowed_imports = sorted(imports - ALLOWED_SOLUTION_IMPORTS)
        network_matches = sorted(set(NETWORK_RE.findall(text)))
        return {
            "exists": path.exists(),
            "imports": sorted(imports),
            "allowed_imports": sorted(ALLOWED_SOLUTION_IMPORTS),
            "disallowed_imports": disallowed_imports,
            "network_call_matches": network_matches,
            "passed": path.exists() and not disallowed_imports and not network_matches,
        }

    def _scan_api_keys(self, run_id: str) -> Dict[str, Any]:
        roots = [
            self.repo_root / "agent" / "archive",
            self.repo_root / "agent_runs" / run_id,
            self.repo_root / "submission",
            self.repo_root / "task_A" / "run_log_a.log",
            self.repo_root / "task_B" / "run_log_b.log",
        ]
        matches: List[Dict[str, str]] = []
        for path in _iter_text_files(roots):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if API_KEY_RE.search(text):
                matches.append({"path": _rel(self.repo_root, path), "kind": "api_key_pattern"})
        return {"roots": [_rel(self.repo_root, path) for path in roots], "matches": matches}


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_diff_protected(repo_root: Path) -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", "--", *PROTECTED_RELATIVE_FILES],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except Exception:
        return ["git_diff_check_failed"]
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _solution_imports(text: str) -> set:
    imports = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"syntax_error"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def _iter_text_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
                    yield path


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path)
