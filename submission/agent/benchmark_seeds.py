"""External benchmark warm-start seeds for circle packing.

The data in benchmarks/dominikkamp is a local, tracked copy of public files
from https://github.com/DominikKamp/Packing. This module never performs network
access; it only turns those public geometry files into ordinary Agent
candidates that still must pass the official evaluators before export.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from geometry_utils import PackingData, TASK_SPECS, summarize_margins, validate_packing


SOURCE_NAME = "DominikKamp/Packing"
SOURCE_URL = "https://github.com/DominikKamp/Packing"
SQUARE_SOURCE_FILE = "square/n26/circlepacking_n26.txt"
RECTANGLE_SOURCE_FILE = "rectangle/n21/rectangle_n21.txt"
SQUARE_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/DominikKamp/Packing/main/"
    "square/n26/circlepacking_n26.txt"
)
RECTANGLE_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/DominikKamp/Packing/main/"
    "rectangle/n21/rectangle_n21.txt"
)
SAFETY_SHRINK = 1.0 - 1e-10
STRATEGY_NAME = "benchmark_seed_dominikkamp"
FICO_SOURCE_NAME = "FICO public newsolutions.txt"
FICO_SOURCE_URL = "https://www.fico.com/en/products/fico-xpress-optimization"


@dataclass
class RawSeed:
    n: int
    raw_sum_radii: float
    triples: np.ndarray
    source_file: str
    upstream_url: str
    local_path: Path


@dataclass
class BenchmarkSeedSolution:
    task: str
    code: str
    data: PackingData
    diagnostics: Dict[str, object]
    source_metadata: Dict[str, object]


def load_dominikkamp_square_n26(repo_root: Optional[Path] = None) -> RawSeed:
    """Load the public n=26 unit-square variable-radius packing seed."""
    path = _benchmarks_dir(repo_root) / "square_n26.txt"
    return _load_raw_seed(
        path=path,
        expected_n=int(TASK_SPECS["B"]["n"]),
        source_file=SQUARE_SOURCE_FILE,
        upstream_url=SQUARE_UPSTREAM_URL,
    )


def load_dominikkamp_rectangle_n21(repo_root: Optional[Path] = None) -> RawSeed:
    """Load the public n=21 perimeter-4 rectangle packing seed."""
    path = _benchmarks_dir(repo_root) / "rectangle_n21.txt"
    return _load_raw_seed(
        path=path,
        expected_n=int(TASK_SPECS["A"]["n"]),
        source_file=RECTANGLE_SOURCE_FILE,
        upstream_url=RECTANGLE_UPSTREAM_URL,
    )


def make_task_b_solution_from_seed(repo_root: Optional[Path] = None) -> BenchmarkSeedSolution:
    """Convert the square n=26 benchmark seed into static Task B solution code."""
    seed = load_dominikkamp_square_n26(repo_root)
    centers = np.asarray(seed.triples[:, :2], dtype=float)
    radii = np.asarray(seed.triples[:, 2], dtype=float) * SAFETY_SHRINK
    data = PackingData(task="B", centers=centers, radii=radii)
    source_metadata = _source_metadata(seed, converted_width_height=None)
    diagnostics = _diagnostics(data, source_metadata)
    code = _solution_code_for(data, diagnostics)
    return BenchmarkSeedSolution(
        task="B",
        code=code,
        data=data,
        diagnostics=diagnostics,
        source_metadata=source_metadata,
    )


def make_task_a_solution_from_seed(repo_root: Optional[Path] = None) -> BenchmarkSeedSolution:
    """Convert the rectangle n=21 benchmark seed into static Task A solution code."""
    seed = load_dominikkamp_rectangle_n21(repo_root)
    raw_centers = np.asarray(seed.triples[:, :2], dtype=float)
    raw_radii = np.asarray(seed.triples[:, 2], dtype=float)
    raw_width = float(np.max(raw_centers[:, 0] + raw_radii))
    raw_height = float(np.max(raw_centers[:, 1] + raw_radii))
    scale = 2.0 / (raw_width + raw_height)

    width = raw_width * scale
    height = raw_height * scale
    centers = raw_centers * scale
    radii = raw_radii * scale * SAFETY_SHRINK
    data = PackingData(task="A", centers=centers, radii=radii, width=width, height=height)
    source_metadata = _source_metadata(
        seed,
        converted_width_height={"width": width, "height": height},
        raw_width_height={"width": raw_width, "height": raw_height},
        scale=scale,
    )
    diagnostics = _diagnostics(data, source_metadata)
    code = _solution_code_for(data, diagnostics)
    return BenchmarkSeedSolution(
        task="A",
        code=code,
        data=data,
        diagnostics=diagnostics,
        source_metadata=source_metadata,
    )


def make_optional_fico_task_a_solution_from_seed(parent_data: Optional[PackingData] = None,
                                                 repo_root: Optional[Path] = None) -> BenchmarkSeedSolution:
    """Load an optional local FICO Problem 13 Task A seed if present.

    The repository does not depend on this public file. If the local copy is not
    available, return the parent geometry unchanged with metadata explaining the
    skipped seed so the Agent run remains reproducible.
    """
    path = _find_fico_seed_path(repo_root)
    if path is None:
        if parent_data is None:
            parent_data = make_task_a_solution_from_seed(repo_root).data
        source_metadata = {
            "source": FICO_SOURCE_NAME,
            "source_url": FICO_SOURCE_URL,
            "source_file": "newsolutions.txt Problem 13",
            "local_path": None,
            "status": "skipped_missing_local_public_seed",
        }
        diagnostics = _diagnostics(parent_data, source_metadata)
        diagnostics["skipped"] = True
        diagnostics["skip_reason"] = "No local FICO newsolutions.txt Problem 13 seed file was found."
        code = _solution_code_for(parent_data, diagnostics, strategy="optional_fico_task_a_seed")
        return BenchmarkSeedSolution(
            task="A",
            code=code,
            data=parent_data,
            diagnostics=diagnostics,
            source_metadata=source_metadata,
        )

    triples, raw_sum = _load_fico_problem13_triples(path)
    raw_centers = triples[:, :2]
    raw_radii = triples[:, 2]
    raw_width = float(np.max(raw_centers[:, 0] + raw_radii))
    raw_height = float(np.max(raw_centers[:, 1] + raw_radii))
    scale = 2.0 / (raw_width + raw_height)
    width = raw_width * scale
    height = raw_height * scale
    centers = raw_centers * scale
    radii = raw_radii * scale * SAFETY_SHRINK
    data = PackingData(task="A", centers=centers, radii=radii, width=width, height=height)
    source_metadata = {
        "source": FICO_SOURCE_NAME,
        "source_url": FICO_SOURCE_URL,
        "source_file": "newsolutions.txt Problem 13",
        "local_path": _relative_seed_path(path, repo_root),
        "raw_sum_radii": raw_sum,
        "safety_shrink": SAFETY_SHRINK,
        "converted_width_height": {"width": width, "height": height},
        "raw_width_height": {"width": raw_width, "height": raw_height},
        "normalization_scale": scale,
        "status": "loaded",
    }
    diagnostics = _diagnostics(data, source_metadata)
    code = _solution_code_for(data, diagnostics, strategy="optional_fico_task_a_seed")
    return BenchmarkSeedSolution(
        task="A",
        code=code,
        data=data,
        diagnostics=diagnostics,
        source_metadata=source_metadata,
    )


def _benchmarks_dir(repo_root: Optional[Path] = None) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent.parent
    return root / "benchmarks" / "dominikkamp"


def _repo_root(repo_root: Optional[Path] = None) -> Path:
    return Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent.parent


def _load_raw_seed(path: Path, expected_n: int, source_file: str, upstream_url: str) -> RawSeed:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark seed file: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Benchmark seed file is incomplete: {path}")
    n = int(lines[0])
    if n != expected_n:
        raise ValueError(f"{path} declares n={n}, expected n={expected_n}")
    raw_sum_radii = float(lines[1])
    triples = np.loadtxt(lines[2:], dtype=float)
    triples = np.asarray(triples, dtype=float).reshape(n, 3)
    return RawSeed(
        n=n,
        raw_sum_radii=raw_sum_radii,
        triples=triples,
        source_file=source_file,
        upstream_url=upstream_url,
        local_path=path,
    )


def _source_metadata(seed: RawSeed, converted_width_height=None,
                     raw_width_height=None, scale: Optional[float] = None) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_file": seed.source_file,
        "upstream_url": seed.upstream_url,
        "local_path": f"benchmarks/dominikkamp/{seed.local_path.name}",
        "raw_sum_radii": float(seed.raw_sum_radii),
        "safety_shrink": SAFETY_SHRINK,
    }
    if converted_width_height is not None:
        metadata["converted_width_height"] = {
            "width": float(converted_width_height["width"]),
            "height": float(converted_width_height["height"]),
        }
    if raw_width_height is not None:
        metadata["raw_width_height"] = {
            "width": float(raw_width_height["width"]),
            "height": float(raw_width_height["height"]),
        }
    if scale is not None:
        metadata["normalization_scale"] = float(scale)
    return metadata


def _diagnostics(data: PackingData, source_metadata: Dict[str, object]) -> Dict[str, object]:
    valid, message = validate_packing(data.task, data.centers, data.radii, data.width, data.height)
    diag: Dict[str, object] = {
        "source": SOURCE_NAME,
        "source_metadata": source_metadata,
        "internal_valid": valid,
        "internal_message": message,
        "sum_radii": data.sum_radii,
        "score_estimate": data.score,
    }
    diag.update(summarize_margins(data.task, data.centers, data.radii, data.width, data.height))
    return diag


def _find_fico_seed_path(repo_root: Optional[Path] = None) -> Optional[Path]:
    root = _repo_root(repo_root)
    candidates = [
        root / "benchmarks" / "fico" / "problem13_task_a.txt",
        root / "benchmarks" / "fico" / "newsolutions.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_fico_problem13_triples(path: Path) -> tuple[np.ndarray, Optional[float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    block = text
    match = re.search(r"(?is)problem\s*13\b(.*?)(?:problem\s*14\b|\Z)", text)
    if match:
        block = match.group(1)
    triples = []
    raw_sum = None
    for line in block.splitlines():
        numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)]
        if raw_sum is None and len(numbers) == 1 and 1.0 < numbers[0] < 4.0:
            raw_sum = numbers[0]
        if len(numbers) >= 3:
            triples.append(numbers[-3:])
    if len(triples) < int(TASK_SPECS["A"]["n"]):
        raise ValueError(f"Could not parse 21 triples from optional FICO seed file: {path}")
    arr = np.asarray(triples[: int(TASK_SPECS["A"]["n"])], dtype=float)
    if raw_sum is None:
        raw_sum = float(np.sum(arr[:, 2]))
    return arr, raw_sum


def _relative_seed_path(path: Path, repo_root: Optional[Path]) -> str:
    root = _repo_root(repo_root)
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _solution_code_for(data: PackingData, diagnostics: Dict[str, object],
                       strategy: str = STRATEGY_NAME) -> str:
    # Reuse the same static solution template used by normal Agent candidates.
    from candidate_generators import solution_code_for

    return solution_code_for(data, strategy=strategy, diagnostics=diagnostics)
