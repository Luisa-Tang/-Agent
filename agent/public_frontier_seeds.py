"""Registry and loader for public frontier seed candidates.

This module records public sources, but only emits seed candidates when raw
coordinates/code are locally available and can be evaluated by the official
scripts. Webpage score claims are metadata only and are never treated as valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from benchmark_seeds import (
    SOURCE_NAME as DOMINIKKAMP_NAME,
    SOURCE_URL as DOMINIKKAMP_URL,
    make_optional_fico_task_a_solution_from_seed,
    make_task_a_solution_from_seed,
    make_task_b_solution_from_seed,
)
from candidate_generators import GeneratedCandidate


RETRIEVAL_TIME_UTC = "2026-07-04T20:30:00Z"


@dataclass
class PublicSeedSource:
    source_id: str
    name: str
    url: str
    retrieval_time_utc: str
    license_note: str
    raw_objective_claim: str
    availability: str
    local_artifacts: List[str]


def source_registry(repo_root: Path) -> List[PublicSeedSource]:
    return [
        PublicSeedSource(
            source_id="dominikkamp_packing",
            name=DOMINIKKAMP_NAME,
            url=DOMINIKKAMP_URL,
            retrieval_time_utc=RETRIEVAL_TIME_UTC,
            license_note="Public GitHub repository; local raw coordinate copies are tracked under benchmarks/dominikkamp with explicit source attribution.",
            raw_objective_claim="rectangle/n21 raw sum 2.365832326862653; square/n26 raw sum 2.635983060895661.",
            availability="coordinates_available_locally",
            local_artifacts=[
                "benchmarks/dominikkamp/rectangle_n21.txt",
                "benchmarks/dominikkamp/square_n26.txt",
            ],
        ),
        PublicSeedSource(
            source_id="claudeevolve_circle_packing",
            name="ClaudeEvolve circle_packing result",
            url="https://github.com/search?q=ClaudeEvolve+circle_packing&type=repositories",
            retrieval_time_utc=RETRIEVAL_TIME_UTC,
            license_note="No local raw coordinates or license file are present in this repository.",
            raw_objective_claim="Not used; no locally available coordinate/code artifact was found.",
            availability="unavailable_no_local_coordinates",
            local_artifacts=[],
        ),
        PublicSeedSource(
            source_id="thetaevolve_circle_packing",
            name="ThetaEvolve circle packing result",
            url="https://github.com/search?q=ThetaEvolve+circle+packing&type=repositories",
            retrieval_time_utc=RETRIEVAL_TIME_UTC,
            license_note="No local raw coordinates or license file are present in this repository.",
            raw_objective_claim="Not used; no locally available coordinate/code artifact was found.",
            availability="unavailable_no_local_coordinates",
            local_artifacts=[],
        ),
        PublicSeedSource(
            source_id="openevolve_issue_156",
            name="OpenEvolve issue #156 code if extractable",
            url="https://github.com/codelion/openevolve/issues/156",
            retrieval_time_utc=RETRIEVAL_TIME_UTC,
            license_note="Issue/code content is not vendored here; no local extractable coordinates were found.",
            raw_objective_claim="Not used as a score claim; only local official evaluator results can validate a seed.",
            availability="unavailable_no_local_extract",
            local_artifacts=[],
        ),
        PublicSeedSource(
            source_id="fico_task_a",
            name="FICO public Task A solution if coordinates are available",
            url="https://www.fico.com/en/products/fico-xpress-optimization",
            retrieval_time_utc=RETRIEVAL_TIME_UTC,
            license_note="Optional public seed support only; no dependency on a network fetch.",
            raw_objective_claim="Only used when a local newsolutions.txt Problem 13 coordinate copy is available.",
            availability="optional_local_file",
            local_artifacts=_fico_local_artifacts(repo_root),
        ),
    ]


def write_source_manifests(repo_root: Path) -> List[Path]:
    paths = []
    root = repo_root / "benchmarks" / "public_frontier"
    for source in source_registry(repo_root):
        directory = root / source.source_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "SOURCE.md"
        path.write_text(_source_markdown(source), encoding="utf-8")
        paths.append(path)
    index = root / "sources.json"
    index.write_text(
        json.dumps([source.__dict__ for source in source_registry(repo_root)], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.append(index)
    return paths


def load_public_frontier_candidates(repo_root: Path, task: str) -> List[GeneratedCandidate]:
    task = task.upper()
    candidates: List[GeneratedCandidate] = []
    if task == "A":
        seed = make_task_a_solution_from_seed(repo_root)
        candidates.append(_to_generated_candidate(seed, "public_frontier_dominikkamp"))
        fico_seed = make_optional_fico_task_a_solution_from_seed(repo_root=repo_root)
        if (fico_seed.source_metadata or {}).get("status") == "loaded":
            candidates.append(_to_generated_candidate(fico_seed, "public_frontier_fico_task_a"))
    elif task == "B":
        seed = make_task_b_solution_from_seed(repo_root)
        candidates.append(_to_generated_candidate(seed, "public_frontier_dominikkamp"))
    return candidates


def frontier_source_summary(repo_root: Path) -> Dict[str, Dict[str, object]]:
    return {source.source_id: source.__dict__ for source in source_registry(repo_root)}


def _to_generated_candidate(seed, strategy: str) -> GeneratedCandidate:
    diagnostics = dict(seed.diagnostics)
    source_metadata = dict(seed.source_metadata)
    source_metadata["frontier_strategy"] = strategy
    diagnostics["source_metadata"] = source_metadata
    diagnostics["public_frontier_seed"] = True
    return GeneratedCandidate(
        task=seed.task,
        strategy=strategy,
        code=seed.code,
        data=seed.data,
        diagnostics=diagnostics,
    )


def _source_markdown(source: PublicSeedSource) -> str:
    lines = [
        f"# {source.name}",
        "",
        f"- Source ID: `{source.source_id}`",
        f"- URL: {source.url}",
        f"- Retrieval time UTC: `{source.retrieval_time_utc}`",
        f"- License note: {source.license_note}",
        f"- Raw objective claim: {source.raw_objective_claim}",
        f"- Availability: `{source.availability}`",
        f"- Local artifacts: `{source.local_artifacts}`",
        "",
        "These files are benchmark warm-start metadata only. A seed can enter the official archive only after the local official `evaluate.py` validates the emitted `solution.py` candidate.",
        "",
    ]
    return "\n".join(lines)


def _fico_local_artifacts(repo_root: Path) -> List[str]:
    root = repo_root / "benchmarks" / "fico"
    if not root.exists():
        return []
    return [str(path.relative_to(repo_root)) for path in root.rglob("*") if path.is_file()]

