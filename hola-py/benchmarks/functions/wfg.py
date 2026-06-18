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

The optimized functions are evaluated through pymoo's WFG implementation so the
landscape matches the Pareto front pymoo generates for the IGD reference (see
``_wfg_pareto_front``). Using the same implementation for both the optimized
function and its reference front keeps them consistent by construction, and
pymoo is already required to generate the reference front.

Our interface accepts position and distance variables ``x0..x_{n-1}`` in
``[0, 1]``; pymoo's WFG expects the i-th decision variable ``z_i`` in
``[0, 2*(i+1)]``, so inputs are rescaled before evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _extract_vec(p: dict[str, float]) -> np.ndarray:
    # Sort by the numeric suffix so x10 follows x9 rather than x1; a plain
    # lexicographic sort would scramble the decision vector at 10+ variables.
    keys = sorted((k for k in p if k.startswith("x")), key=lambda k: int(k[1:]))
    return np.array([p[k] for k in keys])


# Cache pymoo problem instances by (name, n_var, n_obj) so repeated evaluations
# during a benchmark run do not reconstruct the problem on every call.
_WFG_PROBLEM_CACHE: dict[tuple[str, int, int], Any] = {}


def _wfg_eval(name: str, p: dict[str, float], n_obj: int) -> dict[str, float]:
    """Evaluate a WFG problem via pymoo, rescaling our [0, 1] inputs to pymoo's
    ``[0, 2i]`` decision-variable domain."""
    from pymoo.problems import get_problem

    z = _extract_vec(p)
    n_var = len(z)
    key = (name, n_var, n_obj)
    prob = _WFG_PROBLEM_CACHE.get(key)
    if prob is None:
        prob = get_problem(name, n_var=n_var, n_obj=n_obj)
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


def _wfg_pareto_front(name: str, n_obj: int, n_points: int) -> np.ndarray:
    """Generate the true Pareto front for a WFG problem via pymoo."""
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions

    # pymoo uses n_var = k + l where k = n_obj - 1 (position), l = 4 (distance)
    n_var = (n_obj - 1) + 4
    p = get_problem(name, n_var=n_var, n_obj=n_obj)
    try:
        pf = p.pareto_front(n_pareto_points=n_points)
    except TypeError:
        # Some WFG variants (e.g. WFG9) only accept ref_dirs.
        # das-dennis with p partitions in d dims gives C(p+d-1, d-1) points,
        # so choose p to get roughly n_points.
        n_partitions = n_points if n_obj == 2 else 30
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        pf = p.pareto_front(ref_dirs=ref_dirs)
    return np.asarray(pf, dtype=float)


def wfg1_pareto_front(n_obj: int = 2, n_points: int = 500) -> np.ndarray:
    return _wfg_pareto_front("wfg1", n_obj, n_points)


def wfg4_pareto_front(n_obj: int = 2, n_points: int = 500) -> np.ndarray:
    return _wfg_pareto_front("wfg4", n_obj, n_points)


def wfg9_pareto_front(n_obj: int = 2, n_points: int = 500) -> np.ndarray:
    return _wfg_pareto_front("wfg9", n_obj, n_points)
