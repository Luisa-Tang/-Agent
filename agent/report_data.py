"""Generate the submission report from archived agent records."""

from __future__ import annotations

import json
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
    lines.append("- **CandidateGenerator**: `agent/candidate_generators.py` creates standalone solution code from safe grids, staggered starts, SLSQP search, multi-start search, perturb-and-repair, and external benchmark warm-start seeds.")
    lines.append("- **EvaluatorAdapter**: `agent/evaluator_adapter.py` writes candidates into the official task directories and runs `evaluate.py` as the source of truth.")
    lines.append("- **FeedbackReflector**: `agent/run.py` maps failures and plateau behavior to the next strategy; `agent/llm_reflector.py` can optionally ask a compatible Chat Completions endpoint for a strategy suggestion.")
    lines.append("- **ArchiveManager**: `agent/archive.py` stores metadata, raw evaluator output, code snapshots, and best valid candidates.")
    lines.append("- **GeoEvolve-lite**: optional `agent/evolve/` harness evolves candidate-generating programs, novelty filters them, runs cascade evaluation, and exports only official-valid improvements.")
    lines.append("- **Exporter / Reporter**: `agent/run.py` exports final solutions and `agent/report_data.py` creates this report.")
    lines.append("- **Specialist tools**: `agent/skills/` contains project-local reusable procedures for SLSQP search, repair, evaluator feedback, static export, and archive observability.")
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
    lines.append("### Skill-Based Reusable Procedures")
    lines.append("")
    lines.append(
        "The `agent/skills/` layer contains reusable procedures, not personas: "
        "`packing-slsqp`, `packing-repair`, `evaluator-feedback`, `static-export`, "
        "and `archive-observability`. The manager records which procedures were "
        "consulted in each iteration through `skills_used`, so the report can audit "
        "what specialist workflow shaped each candidate."
    )
    lines.append("")
    lines.append("### LangGraph Orchestration Layer")
    lines.append("")
    lines.extend(_langgraph_lines(repo_root))
    lines.append("")
    lines.append("### Explicit Agent State Graph")
    lines.append("")
    lines.extend(_state_graph_lines(repo_root))
    lines.append("")
    lines.append("### Strategy Portfolio Controller")
    lines.append("")
    lines.extend(_strategy_portfolio_lines(repo_root))
    lines.append("")
    lines.append("### Execution Lineage and Replay")
    lines.append("")
    lines.extend(_lineage_lines(repo_root))
    lines.append("")
    lines.append("### Safety Guard and Protected Files")
    lines.append("")
    lines.extend(_safety_lines(repo_root))
    lines.append("")
    lines.append("### Skill Usage Statistics")
    lines.append("")
    lines.extend(_skill_usage_lines(repo_root))
    lines.append("")
    lines.append("### Human-Agent Division Audit")
    lines.append("")
    lines.extend(_human_agent_lines(repo_root))
    lines.append("")
    lines.append("## 3. Code Generation Strategy")
    lines.append("")
    lines.append(
        "The Agent uses template-based generation. Each candidate is a complete "
        "Python module with literal NumPy arrays for centers and radii, plus a "
        "small safety repair routine. Candidate search combines conservative "
        "grid layouts, hexagonal/staggered initializations, SLSQP joint "
        "optimization over centers and radii, multi-start SLSQP, and local "
        "perturbation around the best valid candidate. It can also convert "
        "tracked public benchmark geometry into static candidates and submit "
        "them to the same official evaluator path. For fixed centers, the "
        "Agent solves a linear program to maximize radii under boundary and "
        "pairwise non-overlap constraints, then applies a tiny final shrink/repair."
    )
    lines.append("")
    lines.append(
        "LLM use is optional. When `--use-llm` is passed and `DEEPSEEK_API_KEY` "
        "or a compatible fallback key is set, the Agent asks the configured "
        "Chat Completions-compatible endpoint to "
        "choose among the existing deterministic strategies. The LLM is not allowed "
        "to modify official evaluators, and final exported `solution.py` files remain "
        "standalone with no network or LLM dependency."
    )
    lines.append("")
    lines.append("## 4. External Benchmark Warm-start")
    lines.append("")
    lines.append(
        "The Agent can use public DominikKamp/Packing geometry files as external "
        "benchmark warm-start candidates. This is a seed source inside the Agent "
        "search space, not hidden data and not manually hand-written coordinates. "
        "Each converted seed is emitted as a standalone `solution.py` candidate, "
        "then accepted or rejected only by the official `evaluate.py` scripts."
    )
    lines.append("")
    lines.append("- Source: https://github.com/DominikKamp/Packing")
    lines.append("- Task B seed: `benchmarks/dominikkamp/square_n26.txt` from `square/n26/circlepacking_n26.txt`")
    lines.append("- Task A seed: `benchmarks/dominikkamp/rectangle_n21.txt` from `rectangle/n21/rectangle_n21.txt`")
    lines.append("")
    lines.extend(_external_benchmark_lines(records))
    lines.append("")
    lines.append("### Benchmark-Neighborhood Refinement")
    lines.append("")
    lines.append(
        "After a public benchmark seed is available, the Agent can run three small "
        "neighborhood refinements: fixed-center radius LP, micro center/width "
        "perturbation followed by radius LP, and an optional FICO Problem 13 Task A "
        "seed if a local public copy is available. These are lightweight local "
        "candidate generators, not a new framework. Every candidate still goes "
        "through the official evaluator before it can replace the best valid archive entry."
    )
    lines.append("")
    lines.extend(_refinement_lines(records))
    lines.append("")
    lines.append("### Score Breakthrough Harness")
    lines.append("")
    lines.extend(_breakthrough_lines(repo_root))
    lines.append("")
    lines.append("### Self-Evolution Harness")
    lines.append("")
    lines.extend(_self_evolution_lines(repo_root))
    lines.append("")
    lines.append("## 5. Feedback Utilization")
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
    lines.append("## 6. Termination and Decision Mechanism")
    lines.append("")
    lines.append(
        "The loop terminates after the configured iteration budget or time budget. "
        "The archive keeps every candidate, but only official-evaluator-valid "
        "candidates can become final exports. If no valid optimized candidate is "
        "available, the deterministic safe grid fallback remains valid."
    )
    lines.append("")
    lines.append("## 7. Results")
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
    lines.append("### Skill Usage Summary")
    lines.append("")
    lines.append("| Skill | Iteration uses |")
    lines.append("|---|---:|")
    skill_counts = {}
    for rec in records:
        for skill in rec.get("skills_used") or []:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    for skill, count in sorted(skill_counts.items()):
        lines.append(f"| {skill} | {count} |")
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
    lines.append("## 8. Human-Agent Division")
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
    lines.append("## 9. Limitations and Future Work")
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


def _external_benchmark_lines(records: Iterable[Dict]) -> List[str]:
    benchmark_records = [
        rec for rec in records
        if rec.get("strategy") == "benchmark_seed_dominikkamp"
    ]
    if not benchmark_records:
        return ["No external benchmark warm-start candidate was evaluated in this run."]
    lines = [
        "| Task | Candidate | Source file | Raw sum radii | Official valid | Official score | Official sum radii | Decision |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for rec in benchmark_records:
        source = rec.get("source_metadata") or {}
        official = source.get("official_evaluator_result") or {}
        lines.append(
            f"| {rec.get('task')} | `{rec.get('candidate_id')}` | "
            f"`{source.get('source_file', '')}` | "
            f"{float(source.get('raw_sum_radii') or 0.0):.12f} | "
            f"{official.get('valid')} | "
            f"{float(official.get('score') or 0.0):.6f} | "
            f"{float(official.get('sum_radii') or rec.get('sum_radii') or 0.0):.6f} | "
            f"{rec.get('decision_reason') or rec.get('decision') or ''} |"
        )
    return lines


def _refinement_lines(records: Iterable[Dict]) -> List[str]:
    strategies = {
        "fixed_centers_radius_lp",
        "micro_perturb_lp_refine",
        "optional_fico_task_a_seed",
    }
    refinement_records = [rec for rec in records if rec.get("strategy") in strategies]
    if not refinement_records:
        return ["No benchmark-neighborhood refinement candidate was evaluated in this run."]
    lines = [
        "| Task | Candidate | Strategy | Parent | Valid | Score | Sum radii | Min pairwise margin | Min boundary margin | Failure | Decision |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rec in refinement_records:
        metrics = rec.get("geometry_metrics") or {}
        lines.append(
            f"| {rec.get('task')} | `{rec.get('candidate_id')}` | "
            f"{rec.get('strategy')} | `{rec.get('parent_candidate_id')}` | "
            f"{rec.get('valid')} | "
            f"{float(rec.get('score') or 0.0):.12f} | "
            f"{float(rec.get('sum_radii') or 0.0):.12f} | "
            f"{float(metrics.get('min_pairwise_margin') or 0.0):.3e} | "
            f"{float(metrics.get('min_boundary_margin') or 0.0):.3e} | "
            f"{rec.get('failure_type')} | "
            f"{rec.get('decision_reason') or rec.get('decision') or ''} |"
        )
    return lines


def _breakthrough_lines(repo_root: Path) -> List[str]:
    payload = _load_json(repo_root / "agent" / "archive" / "metrics" / "breakthrough_summary.json")
    if not payload:
        return ["Breakthrough search was not run for this report. Optional mode: `--breakthrough-search`."]
    lines = [
        "The optional breakthrough harness searches near public frontier seeds and contact graph neighborhoods without modifying official evaluators.",
        "- Detailed report: `submission/breakthrough_report.md`",
        "- Candidate log: `agent/archive/metrics/breakthrough_log.jsonl`",
        "- Novelty archive: `agent/archive/metrics/novelty_archive.json`",
        "| Task | Best score | Best sum radii | Gap to 1.0 | Exceeded 1.0 | Generated | Official evals | Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in sorted((payload.get("tasks") or {}).items()):
        best = item.get("best_after") or {}
        lines.append(
            f"| {task} | {float(best.get('score') or 0.0):.12f} | "
            f"{float(best.get('sum_radii') or 0.0):.12f} | "
            f"{float(item.get('gap_to_target') or 0.0):.3e} | "
            f"{item.get('exceeded_target')} | "
            f"{int(item.get('generated_count') or 0)} | "
            f"{int(item.get('official_evaluated_count') or 0)} | "
            f"{int(item.get('valid_count') or 0)} |"
        )
    return lines


def _self_evolution_lines(repo_root: Path) -> List[str]:
    payload = _load_json(repo_root / "agent" / "archive" / "evolve" / "self_evolution_summary.json")
    if not payload:
        return ["Self-evolution search was not run for this report. Optional mode: `--self-evolve-search`."]
    lines = [
        "The optional GeoEvolve-lite harness follows OpenEvolve/ShinkaEvolve/CodeEvolve-inspired ideas without importing those frameworks: a program database, EVOLVE-BLOCK mutations, novelty rejection, cascade evaluation, and an operator bandit.",
        "- Detailed report: `submission/self_evolution_report.md`",
        "- Program DB: `agent/archive/evolve/program_db.jsonl`",
        "- Program tree: `agent/archive/evolve/program_tree.json`",
        "- Evolve log: `agent/archive/evolve/evolve_log.jsonl`",
        "- Operator stats: `agent/archive/evolve/operator_stats.json`",
        "- Block metrics: `agent/archive/evolve/block_metrics.json`",
        "- Evolve Blocks v2 report: `submission/evolve_blocks_v2_report.md`",
        "- Evolve Blocks v3 risky-structure report: `submission/evolve_blocks_v3_risky_structure_report.md`",
        "",
        "Why programs rather than coordinates: the evolved artifact is a small `propose_candidate(parent, rng, context)` generator/refinement operator. Its output is converted to a static candidate and must pass official `evaluate.py` before it can affect final export.",
        "",
        "| Task | Best before | Best after | Improved | Exceeded 1.0 | Gap to 1.0 | Official evals | Valid official |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in sorted((payload.get("tasks") or {}).items()):
        before = item.get("best_before") or {}
        after = item.get("best_after") or {}
        lines.append(
            f"| {task} | {float(before.get('score') or 0.0):.12f} | "
            f"{float(after.get('score') or 0.0):.12f} | "
            f"{item.get('improved_over_start')} | {item.get('exceeded_denominator')} | "
            f"{float(item.get('gap_to_denominator') or 0.0):.3e} | "
            f"{int(item.get('official_evals') or 0)} | {int(item.get('valid_official') or 0)} |"
        )
    lines.extend(
        [
            "",
            f"- Generated programs: `{payload.get('generated_program_count')}`",
            f"- Novelty rejected: `{payload.get('novelty_rejected_count')}`",
            f"- Official evaluate calls: `{payload.get('official_eval_count')}`",
            f"- Accepted improvements: `{payload.get('accepted_improvement_count')}`",
        ]
    )
    risk = payload.get("risk_summary") or {}
    if risk:
        lines.extend(
            [
                "",
                "#### Risky Structure Search",
                "",
                f"- Island counts: `{risk.get('island_counts') or {}}`",
                "| Task | Repair attempted | Repair success | Raw invalid | New contact graphs | New boundary patterns | Best risky delta |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for task, item in sorted((risk.get("tasks") or {}).items()):
            lines.append(
                f"| {task} | {int(item.get('repair_attempted') or 0)} | "
                f"{int(item.get('repair_success') or 0)} | {int(item.get('raw_invalid') or 0)} | "
                f"{int(item.get('new_contact_graph_count') or 0)} | "
                f"{int(item.get('new_boundary_pattern_count') or 0)} | "
                f"{float(item.get('best_risky_candidate_delta') or 0.0):.3e} |"
            )
    operator_stats = payload.get("operator_stats") or {}
    if operator_stats:
        lines.extend(["", "| Operator | Attempts | Valid | Best delta | Novelty mean | Common failures |", "|---|---:|---:|---:|---:|---|"])
        for operator, stat in sorted(operator_stats.items()):
            lines.append(
                f"| `{operator}` | {int(stat.get('attempts') or 0)} | "
                f"{int(stat.get('valid_count') or 0)} | "
                f"{float(stat.get('best_delta') or 0.0):.3e} | "
                f"{float(stat.get('novelty_mean') or 0.0):.3f} | "
                f"`{stat.get('common_failure_types') or {}}` |"
            )
    return lines


def _fmt_optional(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return ""


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rel_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _langgraph_lines(repo_root: Path) -> List[str]:
    mmd = repo_root / "submission" / "demo" / "agent_graph.mmd"
    txt = repo_root / "submission" / "demo" / "agent_graph.txt"
    log_path = repo_root / "agent" / "archive" / "metrics" / "langgraph_run_log.jsonl"
    node_counts: Dict[str, int] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            node = str(record.get("graph_node") or "unknown")
            node_counts[node] = node_counts.get(node, 0) + 1
    lines = [
        "`agent/langgraph_runner.py` is an optional orchestration entrypoint. It wraps the existing deterministic pipeline in a LangGraph `StateGraph` without replacing `agent/run.py`.",
        "State / nodes / conditional edges:",
        "- State: `GeoOptState` records task, iteration, archive summary, strategy stats, selected strategy, evaluator result, best candidate, skills, and artifacts.",
        "- Nodes: `load_task`, `observe_archive`, `select_strategy`, `generate_candidate`, `evaluate_candidate`, `parse_feedback`, `update_archive`, `static_export`, and `safety_check`.",
        "- Conditional edges: after `update_archive`, the graph routes to portfolio selection, direct repair/generation, static export, safety check, or END based on iteration budget, evaluator failure, benchmark availability, plateau, and valid improvement.",
        "Why this is not role-play multi-agent: the graph is one Agent state machine with explicit nodes and deterministic module calls; it does not introduce persona prompts or separate role-playing agents.",
        "Reuse: the LangGraph runner calls the same `EvaluatorAdapter`, `CandidateGenerator`, `ArchiveManager`, `StrategyPortfolioController`, and `SafetyGuard` modules used by the stable pipeline.",
        "Fallback: `agent/run.py` does not import LangGraph and remains runnable when the optional dependency is missing.",
        f"- Mermaid graph: `{_rel_path(repo_root, mmd)}` exists `{mmd.exists()}`.",
        f"- Text graph: `{_rel_path(repo_root, txt)}` exists `{txt.exists()}`.",
        f"- LangGraph node log: `{_rel_path(repo_root, log_path)}` records `{sum(node_counts.values())}` node executions.",
    ]
    if node_counts:
        lines.append("| Graph node | Executions |")
        lines.append("|---|---:|")
        for node, count in sorted(node_counts.items()):
            lines.append(f"| {node} | {count} |")
    return lines


def _state_graph_lines(repo_root: Path) -> List[str]:
    path = repo_root / "agent" / "archive" / "metrics" / "run_log.jsonl"
    if not path.exists():
        return ["No explicit AgentState log has been generated yet."]
    phase_counts: Dict[str, int] = {}
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        phases = [item.get("phase") for item in record.get("state_flow") or [] if item.get("phase")]
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if phases and len(examples) < 3:
            examples.append(f"`{record.get('candidate_id')}`: " + " -> ".join(phases))
    lines = [
        "Each iteration is recorded as `observe -> decide -> act -> evaluate -> archive` using `AgentState` snapshots.",
        "| Phase | Recorded snapshots |",
        "|---|---:|",
    ]
    for phase in ["observe", "decide", "act", "evaluate", "archive"]:
        lines.append(f"| {phase} | {int(phase_counts.get(phase) or 0)} |")
    if examples:
        lines.append("")
        lines.append("Example state paths: " + "; ".join(examples))
    return lines


def _strategy_portfolio_lines(repo_root: Path) -> List[str]:
    payload = _load_json(repo_root / "agent" / "archive" / "metrics" / "strategy_portfolio.json")
    stats = (payload or {}).get("strategy_portfolio_stats") or {}
    if not stats:
        return ["No strategy portfolio metrics have been generated yet."]
    lines = [
        "The controller scores strategies from archive history, evaluator failures, plateau state, score gap, and remaining budget.",
        "| Strategy | Attempts | Validity rate | Best score | Avg score delta | Avg runtime | Common failures | Last used |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for strategy, stat in sorted(stats.items()):
        lines.append(
            f"| {strategy} | {int(stat.get('attempts') or 0)} | "
            f"{float(stat.get('validity_rate') or 0.0):.3f} | "
            f"{float(stat.get('best_score') or 0.0):.6f} | "
            f"{float(stat.get('avg_score_delta') or 0.0):.6g} | "
            f"{float(stat.get('avg_runtime') or 0.0):.3f} | "
            f"`{stat.get('common_failure_types') or {}}` | "
            f"{_fmt_optional(stat.get('last_used_iteration'))} |"
        )
    return lines


def _lineage_lines(repo_root: Path) -> List[str]:
    lines = [
        "Best-candidate lineage DAGs are emitted as replayable JSON. Each node includes parent, strategy, input/output artifacts, code hash, data hash, official score, and decision reason.",
    ]
    for task in ("A", "B"):
        rel = f"agent/archive/lineage/task_{task}_best_lineage.json"
        payload = _load_json(repo_root / rel)
        if not payload:
            lines.append(f"- Task {task}: `{rel}` not generated yet.")
            continue
        lines.append(
            f"- Task {task}: `{rel}` best `{payload.get('best_candidate_id')}`, "
            f"nodes `{len(payload.get('nodes') or [])}`, chain length `{len(payload.get('best_chain') or [])}`."
        )
    return lines


def _safety_lines(repo_root: Path) -> List[str]:
    payload = _load_json(repo_root / "submission" / "safety_report.json") or _load_json(repo_root / "agent" / "archive" / "metrics" / "safety_report.json")
    if not payload:
        return ["Safety guard has not produced a report yet."]
    protected = payload.get("protected_files") or {}
    secret = payload.get("secret_scan") or {}
    lines = [
        f"- Overall safety status: `{payload.get('passed')}`",
        f"- Protected files unchanged: `{protected.get('unchanged')}`",
        f"- Protected git diff entries: `{protected.get('git_diff_protected_files') or []}`",
        f"- API key pattern matches: `{len(secret.get('matches') or [])}`",
    ]
    for path, check in sorted((payload.get("final_solutions") or {}).items()):
        lines.append(
            f"- `{path}` imports `{check.get('imports')}`; network matches `{check.get('network_call_matches')}`; passed `{check.get('passed')}`."
        )
    return lines


def _skill_usage_lines(repo_root: Path) -> List[str]:
    payload = _load_json(repo_root / "agent" / "skills" / "usage_stats.json")
    if not payload:
        return ["No skill usage stats have been generated yet."]
    iterations = payload.get("iterations") or []
    counts: Dict[str, int] = {}
    for item in iterations:
        for skill in item.get("used_skills") or []:
            counts[skill] = counts.get(skill, 0) + 1
    lines = [
        f"- Loaded skills: `{payload.get('loaded_skills') or []}`",
        f"- Iteration records: `{len(iterations)}`",
        "| Skill | Uses |",
        "|---|---:|",
    ]
    for skill, count in sorted(counts.items()):
        lines.append(f"| {skill} | {count} |")
    return lines


def _human_agent_lines(repo_root: Path) -> List[str]:
    payload = _load_json(repo_root / "submission" / "human_agent_division.json")
    if not payload:
        return ["Human-agent division audit has not been generated yet."]
    lines = [
        "Human contributions and Agent actions are audited with evidence artifacts.",
        f"- Human-provided items: `{len(payload.get('human_provided') or [])}`",
        f"- Agent-completed items: `{len(payload.get('agent_completed') or [])}`",
        f"- Audit files: `submission/human_agent_division.md`, `submission/human_agent_division.json`",
    ]
    return lines
