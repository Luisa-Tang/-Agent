"""Deterministic candidate solution generators for both packing tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from benchmark_seeds import (
    STRATEGY_NAME as BENCHMARK_SEED_STRATEGY,
    make_task_a_solution_from_seed,
    make_task_b_solution_from_seed,
    make_optional_fico_task_a_solution_from_seed,
)
from geometry_utils import (
    PackingData,
    TASK_SPECS,
    clip_centers,
    solve_radii_lp,
    structured_initial_data,
    summarize_margins,
    safety_repair,
    validate_packing,
)


STRATEGIES = [
    "baseline_safe_grid",
    "scipy_slsqp_joint",
    "multi_start_slsqp",
    "hexagonal_or_staggered_initialization",
    "perturb_best_and_repair",
    "benchmark_seed_dominikkamp",
    "fixed_centers_radius_lp",
    "micro_perturb_lp_refine",
    "optional_fico_task_a_seed",
]


@dataclass
class GeneratedCandidate:
    task: str
    strategy: str
    code: str
    data: PackingData
    diagnostics: Dict[str, object] = field(default_factory=dict)


class CandidateGenerator:
    def __init__(self, seed: int = 42, fast: bool = False):
        self.seed = int(seed)
        self.fast = bool(fast)

    def generate(self, task: str, strategy: str, iteration: int,
                 parent_data: Optional[PackingData] = None,
                 feedback: Optional[dict] = None) -> GeneratedCandidate:
        task = task.upper()
        if strategy not in STRATEGIES:
            strategy = "baseline_safe_grid"

        if strategy == "baseline_safe_grid":
            data = structured_initial_data(task, "baseline_safe_grid", self.seed + iteration)
            diag = {"source": "safe deterministic grid"}
        elif strategy == "hexagonal_or_staggered_initialization":
            data = self._structured_candidate(task, iteration, "hexagonal_or_staggered_initialization")
            diag = {"source": "staggered rows plus fixed-center LP"}
        elif strategy == "scipy_slsqp_joint":
            init = self._structured_candidate(task, iteration, "random_jittered_grid")
            data, diag = self._optimize_candidate(task, init, maxiter=(160 if self.fast else 450))
        elif strategy == "multi_start_slsqp":
            data, diag = self._multi_start(task, iteration)
        elif strategy == "perturb_best_and_repair":
            if parent_data is None:
                parent_data = self._structured_candidate(task, iteration, "hexagonal_or_staggered_initialization")
            data, diag = self._perturb_and_repair(task, parent_data, iteration)
        elif strategy == BENCHMARK_SEED_STRATEGY:
            seed_solution = (
                make_task_a_solution_from_seed()
                if task == "A"
                else make_task_b_solution_from_seed()
            )
            return GeneratedCandidate(
                task=task,
                strategy=strategy,
                code=seed_solution.code,
                data=seed_solution.data,
                diagnostics=seed_solution.diagnostics,
            )
        elif strategy == "fixed_centers_radius_lp":
            if parent_data is None:
                parent_data = self._structured_candidate(task, iteration, "hexagonal_or_staggered_initialization")
            data, diag = self._fixed_centers_radius_lp(task, parent_data)
        elif strategy == "micro_perturb_lp_refine":
            if parent_data is None:
                parent_data = self._structured_candidate(task, iteration, "hexagonal_or_staggered_initialization")
            data, diag = self._micro_perturb_lp_refine(task, parent_data, iteration)
        elif strategy == "optional_fico_task_a_seed":
            if task == "A":
                seed_solution = make_optional_fico_task_a_solution_from_seed(parent_data=parent_data)
                return GeneratedCandidate(
                    task=task,
                    strategy=strategy,
                    code=seed_solution.code,
                    data=seed_solution.data,
                    diagnostics=seed_solution.diagnostics,
                )
            if parent_data is None:
                parent_data = self._structured_candidate(task, iteration, "hexagonal_or_staggered_initialization")
            data, diag = self._fixed_centers_radius_lp(task, parent_data, source="FICO seed not applicable to Task B")
        else:
            data = structured_initial_data(task, "baseline_safe_grid", self.seed + iteration)
            diag = {"source": "fallback"}

        valid, msg = validate_packing(task, data.centers, data.radii, data.width, data.height)
        diag["internal_valid"] = valid
        diag["internal_message"] = msg
        diag["sum_radii"] = data.sum_radii
        diag["score_estimate"] = data.score
        diag.update(summarize_margins(task, data.centers, data.radii, data.width, data.height))
        code = solution_code_for(data, strategy=strategy, diagnostics=diag)
        return GeneratedCandidate(task=task, strategy=strategy, code=code, data=data, diagnostics=diag)

    def _structured_candidate(self, task: str, iteration: int, variant: str) -> PackingData:
        if task == "A":
            aspects = [1.0, 1.1, 0.9, 1.2, 0.8, 1.35, 0.65, 1.45, 0.55]
            width = aspects[iteration % len(aspects)]
            return structured_initial_data(task, variant, self.seed + 101 * iteration, width_hint=width)
        return structured_initial_data(task, variant, self.seed + 101 * iteration)

    def _multi_start(self, task: str, iteration: int) -> tuple[PackingData, dict]:
        starts = 2 if self.fast else 8
        if task == "A":
            aspects = [1.0, 1.1, 0.9, 1.2, 0.8, 1.35, 0.65, 1.45, 0.55, 1.55, 0.45]
            variants = [
                "hexagonal_or_staggered_initialization",
                "random_jittered_grid",
                "baseline_safe_grid",
            ]
            initial_data = []
            for k in range(starts):
                variant = variants[(iteration + k) % len(variants)]
                width = aspects[(iteration + k) % len(aspects)]
                initial_data.append(structured_initial_data(
                    task, variant, self.seed + 1009 * iteration + k, width_hint=width
                ))
        else:
            variants = [
                "hexagonal_or_staggered_initialization",
                "random_jittered_grid",
                "baseline_safe_grid",
                "auto",
            ]
            initial_data = [
                structured_initial_data(task, variants[(iteration + k) % len(variants)],
                                        self.seed + 1009 * iteration + k)
                for k in range(starts)
            ]

        best_data: Optional[PackingData] = None
        details: List[dict] = []
        for k, init in enumerate(initial_data):
            data, diag = self._optimize_candidate(
                task, init, maxiter=(120 if self.fast else 360), seed_offset=iteration * 37 + k
            )
            details.append(diag)
            if best_data is None or data.sum_radii > best_data.sum_radii:
                best_data = data

        assert best_data is not None
        return best_data, {
            "source": "multi-start SLSQP",
            "starts": starts,
            "start_details": details,
        }

    def _perturb_and_repair(self, task: str, parent_data: PackingData,
                            iteration: int) -> tuple[PackingData, dict]:
        rng = np.random.default_rng(self.seed + 7919 * (iteration + 1))
        centers = np.asarray(parent_data.centers, dtype=float).copy()
        if task == "A":
            width = float(parent_data.width)
            height = 2.0 - width
            scale = min(width, height)
            centers += rng.normal(0.0, 0.015 * scale, size=centers.shape)
            width += float(rng.normal(0.0, 0.025))
            width = float(np.clip(width, 0.35, 1.65))
            height = 2.0 - width
            centers = clip_centers(task, centers, width, height, eps=1e-6)
            radii = solve_radii_lp(task, centers, width, height, margin=1e-7)
            centers, radii = safety_repair(task, centers, radii, width, height)
            init = PackingData(task=task, centers=centers, radii=radii, width=width, height=height)
        else:
            centers += rng.normal(0.0, 0.015, size=centers.shape)
            centers = clip_centers(task, centers, eps=1e-6)
            radii = solve_radii_lp(task, centers, margin=1e-7)
            centers, radii = safety_repair(task, centers, radii)
            init = PackingData(task=task, centers=centers, radii=radii)

        data, diag = self._optimize_candidate(task, init, maxiter=(120 if self.fast else 320),
                                              seed_offset=iteration * 53)
        diag["source"] = "perturbed best valid candidate then SLSQP/LP repair"
        diag["parent_sum_radii"] = parent_data.sum_radii
        return data, diag

    def _fixed_centers_radius_lp(self, task: str, parent_data: PackingData,
                                 source: str = "archive best centers") -> tuple[PackingData, dict]:
        centers = np.asarray(parent_data.centers, dtype=float).copy()
        if task == "A":
            width = float(parent_data.width)
            height = 2.0 - width
            centers = clip_centers(task, centers, width, height, eps=2e-10)
            radii = solve_radii_lp(task, centers, width, height, margin=0.0)
            centers, radii = safety_repair(task, centers, radii, width, height, safety=2e-10)
            data = PackingData(task=task, centers=centers, radii=radii, width=width, height=height)
        else:
            centers = clip_centers(task, centers, eps=2e-10)
            radii = solve_radii_lp(task, centers, margin=0.0)
            centers, radii = safety_repair(task, centers, radii, safety=2e-10)
            data = PackingData(task=task, centers=centers, radii=radii)

        return data, {
            "source": "benchmark-neighborhood refinement",
            "source_metadata": {
                "source": source,
                "source_type": "fixed_centers_radius_lp",
                "parent_sum_radii": parent_data.sum_radii,
                "lp_margin": 0.0,
                "safety_repair": 2e-10,
            },
            "refinement_type": "fixed_centers_radius_lp",
            "parent_sum_radii": parent_data.sum_radii,
            "lp_margin": 0.0,
            "radii_only": True,
        }

    def _micro_perturb_lp_refine(self, task: str, parent_data: PackingData,
                                 iteration: int) -> tuple[PackingData, dict]:
        rng = np.random.default_rng(self.seed + 15485863 * (iteration + 1))
        sigmas = [1e-5, 3e-6, 1e-6, 3e-7]
        trials_per_sigma = 1 if self.fast else 6
        if self.fast:
            sigmas = [3e-6, 1e-6]

        best_data: Optional[PackingData] = None
        best_sigma: Optional[float] = None
        attempts = 0
        width_scale = min(float(parent_data.width or 1.0), float(parent_data.height or 1.0))
        for sigma in sigmas:
            for _ in range(trials_per_sigma):
                attempts += 1
                centers = np.asarray(parent_data.centers, dtype=float).copy()
                centers += rng.normal(0.0, sigma, size=centers.shape)
                if task == "A":
                    width = float(parent_data.width) + float(rng.normal(0.0, sigma * max(width_scale, 1e-6)))
                    width = float(np.clip(width, 0.28, 1.72))
                    height = 2.0 - width
                    centers = clip_centers(task, centers, width, height, eps=2e-10)
                    radii = solve_radii_lp(task, centers, width, height, margin=0.0)
                    centers, radii = safety_repair(task, centers, radii, width, height, safety=2e-10)
                    data = PackingData(task=task, centers=centers, radii=radii, width=width, height=height)
                else:
                    centers = clip_centers(task, centers, eps=2e-10)
                    radii = solve_radii_lp(task, centers, margin=0.0)
                    centers, radii = safety_repair(task, centers, radii, safety=2e-10)
                    data = PackingData(task=task, centers=centers, radii=radii)
                if best_data is None or data.sum_radii > best_data.sum_radii:
                    best_data = data
                    best_sigma = sigma

        assert best_data is not None
        return best_data, {
            "source": "benchmark-neighborhood refinement",
            "source_metadata": {
                "source": "archive best geometry plus micro perturbation",
                "source_type": "micro_perturb_lp_refine",
                "parent_sum_radii": parent_data.sum_radii,
                "sigmas": sigmas,
                "trials_per_sigma": trials_per_sigma,
                "attempts": attempts,
                "best_sigma": best_sigma,
                "lp_margin": 0.0,
                "safety_repair": 2e-10,
            },
            "refinement_type": "micro_perturb_lp_refine",
            "parent_sum_radii": parent_data.sum_radii,
            "sigmas": sigmas,
            "trials_per_sigma": trials_per_sigma,
            "attempts": attempts,
            "best_sigma": best_sigma,
            "lp_margin": 0.0,
        }

    def _optimize_candidate(self, task: str, init: PackingData, maxiter: int,
                            seed_offset: int = 0) -> tuple[PackingData, dict]:
        try:
            from scipy.optimize import minimize
        except Exception as exc:
            return init, {"optimizer": "unavailable", "exception": repr(exc)}

        n = int(TASK_SPECS[task]["n"])
        margin = 2e-7 if self.fast else 8e-8

        if task == "A":
            z0 = np.concatenate([
                np.array([float(init.width)]),
                np.asarray(init.centers, dtype=float).reshape(-1),
                np.asarray(init.radii, dtype=float),
            ])
            bounds = [(0.28, 1.72)] + [(0.0, 1.75)] * (2 * n) + [(0.0, 0.8)] * n

            def unpack(z):
                width = float(z[0])
                height = 2.0 - width
                centers = np.asarray(z[1:1 + 2 * n], dtype=float).reshape(n, 2)
                radii = np.asarray(z[1 + 2 * n:1 + 3 * n], dtype=float)
                return width, height, centers, radii

            def objective(z):
                return -float(np.sum(z[1 + 2 * n:1 + 3 * n]))

            def constraints(z):
                width, height, centers, radii = unpack(z)
                x = centers[:, 0]
                y = centers[:, 1]
                vals = [
                    x - radii - margin,
                    width - x - radii - margin,
                    y - radii - margin,
                    height - y - radii - margin,
                ]
                ii, jj = np.triu_indices(n, 1)
                delta = centers[ii] - centers[jj]
                dist = np.sqrt(np.sum(delta * delta, axis=1) + 1e-18)
                vals.append(dist - radii[ii] - radii[jj] - margin)
                return np.concatenate(vals)

            result_x = z0
            try:
                res = minimize(
                    objective,
                    z0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[{"type": "ineq", "fun": constraints}],
                    options={"maxiter": int(maxiter), "ftol": 1e-9, "disp": False},
                )
                if res.x is not None and np.isfinite(res.x).all():
                    result_x = np.asarray(res.x, dtype=float)
                success = bool(res.success)
                message = str(res.message)
                nit = int(getattr(res, "nit", -1))
            except Exception as exc:
                success = False
                message = repr(exc)
                nit = -1

            width, height, centers, _ = unpack(result_x)
            width = float(np.clip(width, 0.28, 1.72))
            height = 2.0 - width
            centers = clip_centers(task, centers, width, height, eps=1e-7)
            radii = solve_radii_lp(task, centers, width, height, margin=margin)
            centers, radii = safety_repair(task, centers, radii, width, height)
            data = PackingData(task=task, centers=centers, radii=radii, width=width, height=height)
            return data, {
                "optimizer": "SLSQP",
                "success": success,
                "message": message,
                "nit": nit,
                "maxiter": int(maxiter),
                "margin": margin,
                "seed_offset": seed_offset,
            }

        z0 = np.concatenate([
            np.asarray(init.centers, dtype=float).reshape(-1),
            np.asarray(init.radii, dtype=float),
        ])
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

        def unpack_b(z):
            centers = np.asarray(z[:2 * n], dtype=float).reshape(n, 2)
            radii = np.asarray(z[2 * n:3 * n], dtype=float)
            return centers, radii

        def objective_b(z):
            return -float(np.sum(z[2 * n:3 * n]))

        def constraints_b(z):
            centers, radii = unpack_b(z)
            x = centers[:, 0]
            y = centers[:, 1]
            vals = [
                x - radii - margin,
                1.0 - x - radii - margin,
                y - radii - margin,
                1.0 - y - radii - margin,
            ]
            ii, jj = np.triu_indices(n, 1)
            delta = centers[ii] - centers[jj]
            dist = np.sqrt(np.sum(delta * delta, axis=1) + 1e-18)
            vals.append(dist - radii[ii] - radii[jj] - margin)
            return np.concatenate(vals)

        result_x = z0
        try:
            res = minimize(
                objective_b,
                z0,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": constraints_b}],
                options={"maxiter": int(maxiter), "ftol": 1e-9, "disp": False},
            )
            if res.x is not None and np.isfinite(res.x).all():
                result_x = np.asarray(res.x, dtype=float)
            success = bool(res.success)
            message = str(res.message)
            nit = int(getattr(res, "nit", -1))
        except Exception as exc:
            success = False
            message = repr(exc)
            nit = -1

        centers, _ = unpack_b(result_x)
        centers = clip_centers(task, centers, eps=1e-7)
        radii = solve_radii_lp(task, centers, margin=margin)
        centers, radii = safety_repair(task, centers, radii)
        data = PackingData(task=task, centers=centers, radii=radii)
        return data, {
            "optimizer": "SLSQP",
            "success": success,
            "message": message,
            "nit": nit,
            "maxiter": int(maxiter),
            "margin": margin,
            "seed_offset": seed_offset,
        }


def _array_literal(name: str, arr: np.ndarray) -> str:
    body = np.array2string(
        np.asarray(arr, dtype=float),
        precision=17,
        separator=", ",
        max_line_width=120,
    )
    return f"{name} = np.array({body}, dtype=float)"


def solution_code_for(data: PackingData, strategy: str, diagnostics: Optional[dict] = None) -> str:
    task = data.task
    n = int(TASK_SPECS[task]["n"])
    target = float(TASK_SPECS[task]["target"])
    centers_literal = _array_literal("CENTERS", data.centers)
    radii_literal = _array_literal("RADII", data.radii)
    diag = diagnostics or {}
    header = f'''"""Standalone solution generated by the local Agent.

Task: {task}
Strategy: {strategy}
Estimated sum_radii: {data.sum_radii:.12f}
Estimated score denominator: {target:.6f}
Diagnostics: {diag!r}
"""

import numpy as np

EXPECTED_N = {n}
'''
    repair = r'''
def _pair_repair(centers, radii, width, height):
    safety = 2e-10
    centers = np.asarray(centers, dtype=float).copy()
    radii = np.asarray(radii, dtype=float).copy()
    centers[:, 0] = np.clip(centers[:, 0], safety, width - safety)
    centers[:, 1] = np.clip(centers[:, 1], safety, height - safety)
    limits = np.minimum.reduce([
        centers[:, 0],
        width - centers[:, 0],
        centers[:, 1],
        height - centers[:, 1],
    ])
    radii = np.minimum(np.maximum(radii, 0.0), np.maximum(0.0, limits - safety))
    n = len(radii)
    for _ in range(3):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                total = radii[i] + radii[j]
                limit = max(0.0, d - safety)
                if total > limit and total > 0.0:
                    factor = limit / total
                    radii[i] *= factor
                    radii[j] *= factor
                    changed = True
        if not changed:
            break
    return centers, radii


def _fallback_grid(num_circles, width=1.0, height=1.0):
    cols = int(np.ceil(np.sqrt(num_circles * width / max(height, 1e-12))))
    rows = int(np.ceil(num_circles / cols))
    pts = []
    for row in range(rows):
        for col in range(cols):
            if len(pts) >= num_circles:
                break
            pts.append([(col + 0.5) * width / cols, (row + 0.5) * height / rows])
        if len(pts) >= num_circles:
            break
    centers = np.asarray(pts, dtype=float)
    radius = 0.45 * min(width / cols, height / rows)
    radii = np.full(num_circles, radius, dtype=float)
    return centers, radii
'''
    if task == "A":
        return header + f"\nWIDTH = {float(data.width):.17g}\nHEIGHT = {float(data.height):.17g}\n" + centers_literal + "\n" + radii_literal + repair + r'''


def run_packing(num_circles):
    if int(num_circles) != EXPECTED_N:
        width = 1.0
        height = 1.0
        centers, radii = _fallback_grid(int(num_circles), width, height)
        centers, radii = _pair_repair(centers, radii, width, height)
        return centers, radii, float(width), float(height)
    centers, radii = _pair_repair(CENTERS, RADII, WIDTH, HEIGHT)
    return centers, radii, float(WIDTH), float(HEIGHT)
'''
    return header + centers_literal + "\n" + radii_literal + repair + r'''


def run_packing(num_circles):
    if int(num_circles) != EXPECTED_N:
        centers, radii = _fallback_grid(int(num_circles), 1.0, 1.0)
        centers, radii = _pair_repair(centers, radii, 1.0, 1.0)
        return centers, radii, float(np.sum(radii))
    centers, radii = _pair_repair(CENTERS, RADII, 1.0, 1.0)
    return centers, radii, float(np.sum(radii))
'''
