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

"""DTLZ multi-objective test functions.

Reference: Deb, Thiele, Laumanns & Zitzler (2005), "Scalable Test Problems
for Evolutionary Multiobjective Optimization."

Decision variables x0..x_{n-1} in [0, 1]. Number of objectives M is
configurable. Standard choices: n = M + k where k = 4 (DTLZ1) or k = 9.
"""

from __future__ import annotations

import math

import numpy as np


def _extract_vec(p: dict[str, float]) -> list[float]:
    keys = sorted(k for k in p if k.startswith("x"))
    return [p[k] for k in keys]


def dtlz1(p: dict[str, float], n_obj: int = 3) -> dict[str, float]:
    """DTLZ1: linear Pareto front (hyperplane). k = n - M + 1 = 4 standard."""
    x = _extract_vec(p)
    n = len(x)
    k = n - n_obj + 1
    xm = x[n_obj - 1 :]  # last k variables

    g = 100 * (k + sum((xi - 0.5) ** 2 - math.cos(20 * math.pi * (xi - 0.5)) for xi in xm))

    objs = {}
    for i in range(n_obj):
        f = 0.5 * (1 + g)
        for j in range(n_obj - 1 - i):
            f *= x[j]
        if i > 0:
            f *= 1 - x[n_obj - 1 - i]
        objs[f"f{i + 1}"] = f
    return objs


def dtlz2(p: dict[str, float], n_obj: int = 3) -> dict[str, float]:
    """DTLZ2: spherical Pareto front. k = n - M + 1 = 9 standard."""
    x = _extract_vec(p)
    xm = x[n_obj - 1 :]

    g = sum((xi - 0.5) ** 2 for xi in xm)

    objs = {}
    for i in range(n_obj):
        f = 1 + g
        for j in range(n_obj - 1 - i):
            f *= math.cos(x[j] * math.pi / 2)
        if i > 0:
            f *= math.sin(x[n_obj - 1 - i] * math.pi / 2)
        objs[f"f{i + 1}"] = f
    return objs


def dtlz3(p: dict[str, float], n_obj: int = 3) -> dict[str, float]:
    """DTLZ3: spherical front with multimodal g (many local fronts). k = 9 standard."""
    x = _extract_vec(p)
    n = len(x)
    k = n - n_obj + 1
    xm = x[n_obj - 1 :]

    g = 100 * (k + sum((xi - 0.5) ** 2 - math.cos(20 * math.pi * (xi - 0.5)) for xi in xm))

    objs = {}
    for i in range(n_obj):
        f = 1 + g
        for j in range(n_obj - 1 - i):
            f *= math.cos(x[j] * math.pi / 2)
        if i > 0:
            f *= math.sin(x[n_obj - 1 - i] * math.pi / 2)
        objs[f"f{i + 1}"] = f
    return objs


def dtlz4(p: dict[str, float], n_obj: int = 3, alpha: float = 100.0) -> dict[str, float]:
    """DTLZ4: spherical front with biased density. k = 9 standard."""
    x = _extract_vec(p)
    xm = x[n_obj - 1 :]

    g = sum((xi - 0.5) ** 2 for xi in xm)

    objs = {}
    for i in range(n_obj):
        f = 1 + g
        for j in range(n_obj - 1 - i):
            f *= math.cos(x[j] ** alpha * math.pi / 2)
        if i > 0:
            f *= math.sin(x[n_obj - 1 - i] ** alpha * math.pi / 2)
        objs[f"f{i + 1}"] = f
    return objs


# ---------------------------------------------------------------------------
# True Pareto fronts
# ---------------------------------------------------------------------------


def dtlz1_pareto_front(n_obj: int = 3, n_points: int = 10000) -> np.ndarray:
    """DTLZ1 Pareto front: linear hyperplane sum(f_i) = 0.5."""
    rng = np.random.default_rng(42)
    # Sample uniformly on the simplex
    raw = rng.exponential(1.0, size=(n_points, n_obj))
    front = 0.5 * raw / raw.sum(axis=1, keepdims=True)
    return front


def dtlz2_pareto_front(n_obj: int = 3, n_points: int = 10000) -> np.ndarray:
    """DTLZ2/3/4 Pareto front: unit sphere in first quadrant, sum(f_i^2) = 1."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(size=(n_points, n_obj))
    raw = np.abs(raw)
    front = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return front
