"""Generate the submission report from archived agent records."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from geometry_utils import TASK_SPECS


def generate_report(repo_root: Path, archive, output_path: Path,
                    evaluate_all_output: Optional[str] = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = archive.records
    best_a = archive.best_record("A")
    best_b = archive.best_record("B")
    score_a = float(best_a.get("score") or 0.0) if best_a else 0.0
    score_b = float(best_b.get("score") or 0.0) if best_b else 0.0
    combined = (score_a + score_b) / 2.0

    lines: List[str] = []
    lines.append("# AlgorithmOptimization Local Agent Report")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(
        "This repository contains a deterministic local Agent system that generates "
        "`solution.py` candidates for two circle-packing optimization tasks, runs "
        "the official evaluators through subprocesses, parses feedback, archives "
        "every attempt, and exports the best valid candidates. The harness follows "
        "an observe -> think -> act -> observe loop so every code-evolution step is traceable."
    )
    lines.append("")
    lines.append("## 2. System Architecture")
    lines.append("")
    lines.append("- **ProblemParser**: `agent/run.py` reads task descriptions and evaluator files before the loop starts.")
    lines.append("- **CandidateGenerator**: `agent/candidate_generators.py` creates standalone solution code from safe grids, staggered starts, SLSQP search, multi-start search, and perturb-and-repair.")
    lines.append("- **EvaluatorAdapter**: `agent/evaluator_adapter.py` writes candidates into the official task directories and runs `evaluate.py` as the source of truth.")
    lines.append("- **FeedbackReflector**: `agent/run.py` maps failures and plateau behavior to the next strategy; `agent/llm_reflector.py` can optionally ask a compatible Chat Completions endpoint for a strategy suggestion.")
    lines.append("- **ArchiveManager**: `agent/archive.py` stores metadata, raw evaluator output, code snapshots, and best valid candidates.")
    lines.append("- **Exporter / Reporter**: `agent/run.py` exports final solutions and `agent/report_data.py` creates this report.")
    lines.append("- **Specialist tools**: `skills/` documents SLSQP search, repair, evaluator feedback, and static export skills used by the manager loop.")
    lines.append("")
    lines.append("### Workflow vs Agent")
    lines.append("")
    lines.append(
        "The workflow is the fixed reproducible harness: parse context, generate a candidate, run official evaluators, archive, and export. "
        "The Agent layer is the decision policy on top of that workflow: it observes evaluator/archives, chooses a specialist strategy, acts by generating code, and observes official feedback before continuing."
    )
    lines.append("")
    lines.append("### Observe -> Think -> Act -> Observe Loop")
    lines.append("")
    lines.append("- **Observe**: read the previous evaluator result, best archive state, no-improvement count, and strategy statistics.")
    lines.append("- **Think**: select a strategy with deterministic policy or optional LLM reflection constrained to the strategy whitelist.")
    lines.append("- **Act**: generate standalone `solution.py` code using a specialist optimizer/repair/export operation.")
    lines.append("- **Observe**: run the official evaluator, classify failure, compute geometry metrics, and write the trace to JSONL/logs.")
    lines.append("")
    lines.append("### Manager + Specialist Tools Pattern")
    lines.append("")
    lines.append(
        "`agent/run.py` is the manager. It delegates to specialist tools: SLSQP candidate generation, fixed-center LP repair, evaluator feedback parsing, archive/statistics memory, and static solution export. "
        "The evaluator governs code evolution: only candidates accepted by official scripts can become final exports."
    )
    lines.append("")
    lines.append("## 3. Code Generation Strategy")
    lines.append("")
    lines.append(
        "The Agent uses template-based generation. Each candidate is a complete "
        "Python module with literal NumPy arrays for centers and radii, plus a "
        "small safety repair routine. Candidate search combines conservative "
        "grid layouts, hexagonal/staggered initializations, SLSQP joint "
        "optimization over centers and radii, multi-start SLSQP, and local "
        "perturbation around the best valid candidate. For fixed centers, the "
        "Agent solves a linear program to maximize radii under boundary and "
        "pairwise non-overlap constraints, then applies a tiny final shrink/repair."
    )
    lines.append("")
    lines.append(
        "LLM use is optional. When `--use-llm` is passed and `OPENAI_API_KEY` is "
        "set, the Agent asks a configured Chat Completions-compatible endpoint to "
        "choose among the existing deterministic strategies. The LLM is not allowed "
        "to modify official evaluators, and final exported `solution.py` files remain "
        "standalone with no network or LLM dependency."
    )
    lines.append("")
    lines.append("## 4. Feedback Utilization")
    lines.append("")
    lines.append(
        "Evaluator output is parsed for score, `sum_radii`, validity, and failure "
        "type. Overlap failures trigger more conservative repair. Outside-boundary "
        "failures trigger boundary-tight generation. Low but valid scores move the "
        "Agent toward multi-start and structured initializations. Plateaued valid "
        "runs trigger perturb-and-repair around the current best candidate."
    )
    lines.append("")
    lines.append(
        "Failure classification uses the explicit taxonomy: `shape_error`, `nonfinite`, `negative_radius`, `perimeter_error`, "
        "`boundary_violation`, `overlap`, `timeout`, `low_score`, `plateau`, and `unknown`."
    )
    lines.append("")
    lines.append("## 5. Termination and Decision Mechanism")
    lines.append("")
    lines.append(
        "The loop terminates after the configured iteration budget or time budget. "
        "The archive keeps every candidate, but only official-evaluator-valid "
        "candidates can become final exports. If no valid optimized candidate is "
        "available, the deterministic safe grid fallback remains valid."
    )
    lines.append("")
    lines.append("## 6. Results")
    lines.append("")
    lines.extend(_best_lines("A", best_a))
    lines.extend(_best_lines("B", best_b))
    lines.append(f"- Combined best-valid score: `{combined:.6f}`")
    lines.append("")
    lines.append("### Iteration Trajectory")
    lines.append("")
    lines.append("| Task | Iteration | Candidate | Strategy | Valid | Score | Sum radii | Failure |")
    lines.append("|---|---:|---|---|---:|---:|---:|---|")
    for rec in records:
        metrics = rec.get("geometry_metrics") or {}
        lines.append(
            f"| {rec.get('task')} | {rec.get('iteration')} | `{rec.get('candidate_id')}` | "
            f"{rec.get('strategy')} | {rec.get('valid')} | "
            f"{float(rec.get('score') or 0.0):.6f} | {float(rec.get('sum_radii') or 0.0):.6f} | "
            f"{rec.get('failure_type')} |"
        )
    lines.append("")
    lines.append("### Strategy Archive Statistics")
    lines.append("")
    lines.append("| Strategy | Attempts | Validity rate | Best score | Avg score improvement |")
    lines.append("|---|---:|---:|---:|---:|")
    for strategy, stat in sorted(archive.strategy_stats().items()):
        lines.append(
            f"| {strategy} | {int(stat.get('attempts') or 0)} | "
            f"{float(stat.get('validity_rate') or 0.0):.3f} | "
            f"{float(stat.get('best_score') or 0.0):.6f} | "
            f"{float(stat.get('average_score_improvement') or 0.0):.6f} |"
        )
    lines.append("")
    lines.append("### Best Geometry Safety Metrics")
    lines.append("")
    lines.append("| Task | Min pairwise margin | Min boundary margin | Sum radii | Score | Width | Height |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for task, rec in (("A", best_a), ("B", best_b)):
        metrics = (rec or {}).get("geometry_metrics") or {}
        lines.append(
            f"| {task} | {float(metrics.get('min_pairwise_margin') or 0.0):.3e} | "
            f"{float(metrics.get('min_boundary_margin') or 0.0):.3e} | "
            f"{float((rec or {}).get('sum_radii') or 0.0):.6f} | "
            f"{float((rec or {}).get('score') or 0.0):.6f} | "
            f"{_fmt_optional(metrics.get('width'))} | {_fmt_optional(metrics.get('height'))} |"
        )
    lines.append("")
    if evaluate_all_output:
        lines.append("### Final Evaluator Output")
        lines.append("")
        lines.append("```text")
        lines.append(evaluate_all_output.strip())
        lines.append("```")
        lines.append("")
    lines.append("## 7. Human-Agent Division")
    lines.append("")
    lines.append(
        "The human provided the high-level system design, constraints, required "
        "interfaces, and quality bar. The Agent implemented the framework, ran "
        "the local evaluator loop, generated candidate solution files, selected "
        "the best valid candidates, exported artifacts, and produced logs. Optional "
        "LLM connectivity is supported but the recorded run may use deterministic "
        "fallback unless `--use-llm` and credentials are provided. The main manual "
        "assumption was using the local conda Python environment when the system "
        "`python` command was unavailable."
    )
    lines.append("")
    lines.append("## 8. Limitations and Future Work")
    lines.append("")
    lines.append("- Add broader global search and population-based evolution.")
    lines.append("- Add more diverse initialization families and symmetry-breaking operators.")
    lines.append("- Use LLM-guided operator generation when API access is configured, while keeping deterministic fallback.")
    lines.append("- Add visualization-based diagnosis for overlap and wasted-space patterns.")
    lines.append("- Run longer non-fast searches for higher scores.")
    lines.append("")
    lines.append("## PDF Conversion")
    lines.append("")
    lines.append("If a PDF is required, convert this Markdown file with a local tool such as:")
    lines.append("")
    lines.append("```bash")
    lines.append("pandoc submission/report.md -o submission/report.pdf")
    lines.append("```")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _best_lines(task: str, record: Optional[Dict]) -> List[str]:
    target = TASK_SPECS[task]["target"]
    if not record:
        return [f"- Task {task}: no valid candidate found. Target denominator `{target:.6f}`."]
    lines = [
        f"- Task {task} best candidate: `{record.get('candidate_id')}`",
        f"- Task {task} sum_radii: `{float(record.get('sum_radii') or 0.0):.6f}`",
        f"- Task {task} score: `{float(record.get('score') or 0.0):.6f}` using denominator `{target:.6f}`",
    ]
    if task == "A":
        lines.append(f"- Task A width/height: `{float(record.get('width') or 0.0):.9f}` / `{float(record.get('height') or 0.0):.9f}`")
    return lines


def _fmt_optional(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return ""
