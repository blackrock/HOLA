# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WFG multi-objective test functions.

Reference: Huband, Hingston, Barone & While (2006), "A Review of
Multiobjective Test Problems and a Scalable Test Problem Toolkit."

The optimized functions and deterministic IGD reference fronts are both
evaluated through pymoo's WFG implementation (see ``_wfg_pareto_front``).
Using the same implementation for both keeps its floating-point behavior
consistent by construction.

Our interface accepts position and distance variables ``x0..x_{n-1}`` in
``[0, 1]``; pymoo's WFG expects the i-th decision variable ``z_i`` in
``[0, 2*(i+1)]``, so inputs are rescaled before evaluation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

WFG_REFERENCE_FRONT_VERSION = "das-dennis-direction-mapped-v1"


def _wfg_parameters(n_obj: int) -> tuple[int, int]:
    """Return the standard Huband-recommended position/distance dimensions."""
    if n_obj < 2:
        raise ValueError("WFG problems must have at least two objectives")
    k = 4 if n_obj == 2 else 2 * (n_obj - 1)
    return k, 20


def _extract_vec(p: dict[str, float]) -> np.ndarray:
    # Sort by the numeric suffix so x10 follows x9 rather than x1; a plain
    # lexicographic sort would scramble the decision vector at 10+ variables.
    keys = sorted((k for k in p if k.startswith("x")), key=lambda k: int(k[1:]))
    return np.array([p[k] for k in keys])


# Cache pymoo problem instances by (name, n_var, n_obj, k) so repeated
# evaluations during a benchmark run do not reconstruct the problem on every
# call.
_WFG_PROBLEM_CACHE: dict[tuple[str, int, int, int], Any] = {}


def _wfg_eval(name: str, p: dict[str, float], n_obj: int) -> dict[str, float]:
    """Evaluate a WFG problem via pymoo, rescaling our [0, 1] inputs to pymoo's
    ``[0, 2i]`` decision-variable domain."""
    from pymoo.problems import get_problem

    z = _extract_vec(p)
    n_var = len(z)
    k, distance_dimensions = _wfg_parameters(n_obj)
    expected_n_var = k + distance_dimensions
    if n_var != expected_n_var:
        raise ValueError(
            f"{name.upper()} with {n_obj} objectives requires k={k}, "
            f"l={distance_dimensions}, "
            f"and therefore {expected_n_var} variables; got {n_var}"
        )
    key = (name, n_var, n_obj, k)
    prob = _WFG_PROBLEM_CACHE.get(key)
    if prob is None:
        prob = get_problem(name, n_var=n_var, n_obj=n_obj, k=k, l=distance_dimensions)
        _WFG_PROBLEM_CACHE[key] = prob

    scale = np.array([2.0 * (i + 1) for i in range(n_var)])
    f = prob.evaluate((z * scale).reshape(1, -1))[0]
    return {f"f{m + 1}": float(f[m]) for m in range(n_obj)}


def wfg1(p: dict[str, float], n_obj: int = 2) -> dict[str, float]:
    """WFG1: mixed convex/concave, biased, separable."""
    return _wfg_eval("wfg1", p, n_obj)


def wfg4(p: dict[str, float], n_obj: int = 2) -> dict[str, float]:
    """WFG4: multimodal, separable."""
    return _wfg_eval("wfg4", p, n_obj)


def wfg9(p: dict[str, float], n_obj: int = 2) -> dict[str, float]:
    """WFG9: non-separable, deceptive."""
    return _wfg_eval("wfg9", p, n_obj)


# ---------------------------------------------------------------------------
# Pareto front generators
# ---------------------------------------------------------------------------


def _wfg_reference_directions(n_obj: int, n_points: int) -> np.ndarray:
    """Return a deterministic Das--Dennis grid near the requested cardinality.

    A two-objective grid with ``p`` partitions contains ``p + 1`` directions,
    so the requested cardinality is exact.  A three-objective grid contains
    ``(p + 1) * (p + 2) / 2`` directions; the closest grid is used, with the
    smaller grid winning a tie.  Thus the configured request of 500 points
    produces 500 directions in two dimensions and 496 directions (30
    partitions) in three dimensions.
    """
    from pymoo.util.ref_dirs import get_reference_directions

    if n_obj not in {2, 3}:
        raise ValueError("configured WFG references support two or three objectives")
    if n_points < n_obj:
        raise ValueError(f"a {n_obj}-objective WFG reference needs at least {n_obj} points")

    if n_obj == 2:
        n_partitions = n_points - 1
    else:
        # Solve (p + 1)(p + 2)/2 ~= n_points and compare the adjacent integer
        # partition counts.  Avoid pymoo's stochastic "energy" directions.
        approximate = (math.sqrt(8 * n_points + 1) - 3) / 2
        candidates = {max(1, math.floor(approximate)), max(1, math.ceil(approximate))}
        n_partitions = min(
            candidates,
            key=lambda partitions: (
                abs(math.comb(partitions + 2, 2) - n_points),
                math.comb(partitions + 2, 2),
            ),
        )

    return np.asarray(
        get_reference_directions(
            "das-dennis",
            n_obj,
            n_partitions=n_partitions,
        ),
        dtype=float,
    )


def _bisect_unit_interval(
    residual: Any,
    n_values: int,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Solve bracketed sign-changing residuals in parallel without randomness."""
    low = np.full(n_values, lower, dtype=float)
    high = np.full(n_values, upper, dtype=float)
    for _ in range(64):
        midpoint = (low + high) / 2.0
        negative = np.asarray(residual(midpoint)) < 0.0
        low = np.where(negative, midpoint, low)
        high = np.where(negative, high, midpoint)
    return (low + high) / 2.0


def _wfg1_position_coordinates(ref_dirs: np.ndarray) -> np.ndarray:
    """Map objective-space directions to WFG1's exact Pareto-set coordinates."""
    n_points, n_obj = ref_dirs.shape
    if n_obj == 2:

        def residual(x: np.ndarray) -> np.ndarray:
            convex = 1.0 - np.cos(0.5 * np.pi * x)
            mixed = 1.0 - x - np.cos(10.0 * np.pi * x + 0.5 * np.pi) / (10.0 * np.pi)
            return ref_dirs[:, 1] * convex - ref_dirs[:, 0] * mixed

        x = _bisect_unit_interval(residual, n_points)
        x[ref_dirs[:, 0] == 0.0] = 0.0
        x[ref_dirs[:, 1] == 0.0] = 1.0
        return x[:, None]

    # In three objectives, the first two convex shapes share the factor
    # 1-cos(pi*x1/2).  Solve their ratio for x2, then their joint magnitude
    # relative to the mixed third shape for x1.
    direction_12 = np.linalg.norm(ref_dirs[:, :2], axis=1)

    def second_residual(x: np.ndarray) -> np.ndarray:
        first_shape = 1.0 - np.cos(0.5 * np.pi * x)
        second_shape = 1.0 - np.sin(0.5 * np.pi * x)
        return ref_dirs[:, 1] * first_shape - ref_dirs[:, 0] * second_shape

    x2 = _bisect_unit_interval(second_residual, n_points)
    x2[ref_dirs[:, 0] == 0.0] = 0.0
    x2[ref_dirs[:, 1] == 0.0] = 1.0
    first_shape = 1.0 - np.cos(0.5 * np.pi * x2)
    second_shape = 1.0 - np.sin(0.5 * np.pi * x2)
    shape_12 = np.hypot(first_shape, second_shape)

    def first_residual(x: np.ndarray) -> np.ndarray:
        convex = 1.0 - np.cos(0.5 * np.pi * x)
        mixed = 1.0 - x - np.cos(10.0 * np.pi * x + 0.5 * np.pi) / (10.0 * np.pi)
        return ref_dirs[:, 2] * convex * shape_12 - direction_12 * mixed

    x1 = _bisect_unit_interval(first_residual, n_points)
    x1[direction_12 == 0.0] = 0.0
    x1[ref_dirs[:, 2] == 0.0] = 1.0
    return np.column_stack([x1, x2])


def _concave_position_coordinates(ref_dirs: np.ndarray) -> np.ndarray:
    """Invert WFG's concave shape onto its one- or two-coordinate Pareto set."""
    unit = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    if ref_dirs.shape[1] == 2:
        return (2.0 / np.pi * np.arctan2(unit[:, 0], unit[:, 1]))[:, None]
    return np.column_stack(
        [
            2.0 / np.pi * np.arccos(unit[:, 2]),
            2.0 / np.pi * np.arctan2(unit[:, 0], unit[:, 1]),
        ]
    )


def _evaluate_wfg1_front(problem: Any, ref_dirs: np.ndarray) -> np.ndarray:
    """Evaluate exact WFG1 Pareto-set decisions through the optimized problem."""
    positions = _wfg1_position_coordinates(ref_dirs)
    # WFG1 raises each positional decision to 0.02.  Repeating x**50 within
    # each positional block makes the transformed weighted average exactly x.
    raw_positions = np.power(positions, 50.0)
    gap = problem.k // (problem.n_obj - 1)
    pareto_positions = np.repeat(raw_positions, gap, axis=1)
    pareto_set = problem._positional_to_optimal(pareto_positions)
    return np.asarray(problem.evaluate(pareto_set, return_values_of=["F"]), dtype=float)


def _evaluate_wfg4_front(problem: Any, ref_dirs: np.ndarray) -> np.ndarray:
    """Evaluate exact WFG4 Pareto-set decisions through the optimized problem."""
    positions = _concave_position_coordinates(ref_dirs)
    raw_columns = []
    for column in positions.T:

        def residual(raw: np.ndarray, target: np.ndarray = column) -> np.ndarray:
            # WFG4's multimodal transform is continuous between 0 and 0.35,
            # with values 1 and 0 at those endpoints.  Negate it to match the
            # sign convention used by the parallel bisection helper.
            transformed = type(problem).t1(raw[:, None])[:, 0]
            return target - transformed

        raw_columns.append(
            _bisect_unit_interval(
                residual,
                len(column),
                lower=0.0,
                upper=0.35,
            )
        )

    raw_positions = np.column_stack(raw_columns)
    gap = problem.k // (problem.n_obj - 1)
    pareto_positions = np.repeat(raw_positions, gap, axis=1)
    pareto_set = problem._positional_to_optimal(pareto_positions)
    return np.asarray(problem.evaluate(pareto_set, return_values_of=["F"]), dtype=float)


def _wfg_pareto_front(name: str, n_obj: int, n_points: int) -> np.ndarray:
    """Generate a deterministic direction-matched true WFG Pareto front.

    pymoo's WFG1 and WFG4 front methods ignore ``n_pareto_points`` and sample
    entropy-seeded Pareto-set candidates.  Instead, this maps an explicit
    Das--Dennis direction grid to exact Pareto-set decisions and evaluates them
    through the same pymoo problem used by the benchmark.  WFG9 already has an
    exact deterministic implementation, so its explicit directions are passed
    directly to pymoo.
    """
    from pymoo.problems import get_problem

    k, distance_dimensions = _wfg_parameters(n_obj)
    n_var = k + distance_dimensions
    p = get_problem(name, n_var=n_var, n_obj=n_obj, k=k, l=distance_dimensions)
    ref_dirs = _wfg_reference_directions(n_obj, n_points)
    if name == "wfg1":
        return _evaluate_wfg1_front(p, ref_dirs)
    if name == "wfg4":
        return _evaluate_wfg4_front(p, ref_dirs)
    if name == "wfg9":
        return np.asarray(
            p.pareto_front(ref_dirs=ref_dirs, use_cache=False),
            dtype=float,
        )
    raise ValueError(f"unsupported WFG reference variant: {name}")


def wfg1_pareto_front(n_obj: int = 2, n_points: int = 500) -> np.ndarray:
    return _wfg_pareto_front("wfg1", n_obj, n_points)


def wfg4_pareto_front(n_obj: int = 2, n_points: int = 500) -> np.ndarray:
    return _wfg_pareto_front("wfg4", n_obj, n_points)


def wfg9_pareto_front(n_obj: int = 2, n_points: int = 500) -> np.ndarray:
    return _wfg_pareto_front("wfg9", n_obj, n_points)
