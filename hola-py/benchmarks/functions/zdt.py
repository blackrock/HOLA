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

"""ZDT multi-objective test functions.

Reference: Zitzler, Deb & Thiele (2000), "Comparison of Multiobjective
Evolutionary Algorithms: Empirical Results."

All functions have decision variables x0..x_{n-1} in [0, 1] and two
objectives (f1, f2) to minimize.
"""

from __future__ import annotations

import math

import numpy as np


def _extract_vec(p: dict[str, float]) -> list[float]:
    keys = sorted(k for k in p if k.startswith("x"))
    return [p[k] for k in keys]


def zdt1(p: dict[str, float]) -> dict[str, float]:
    """ZDT1: convex Pareto front. n=30 typical."""
    x = _extract_vec(p)
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    f2 = g * (1 - math.sqrt(f1 / g))
    return {"f1": f1, "f2": f2}


def zdt2(p: dict[str, float]) -> dict[str, float]:
    """ZDT2: non-convex (concave) Pareto front. n=30 typical."""
    x = _extract_vec(p)
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    f2 = g * (1 - (f1 / g) ** 2)
    return {"f1": f1, "f2": f2}


def zdt3(p: dict[str, float]) -> dict[str, float]:
    """ZDT3: disconnected Pareto front. n=30 typical."""
    x = _extract_vec(p)
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    f2 = g * (1 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10 * math.pi * f1))
    return {"f1": f1, "f2": f2}


def zdt4(p: dict[str, float]) -> dict[str, float]:
    """ZDT4: multimodal (many local fronts). n=10 typical, x0 in [0,1], rest in [-5,5]."""
    x = _extract_vec(p)
    n = len(x)
    f1 = x[0]
    g = 1 + 10 * (n - 1) + sum(xi**2 - 10 * math.cos(4 * math.pi * xi) for xi in x[1:])
    f2 = g * (1 - math.sqrt(f1 / g))
    return {"f1": f1, "f2": f2}


def zdt6(p: dict[str, float]) -> dict[str, float]:
    """ZDT6: non-uniform, biased. n=10 typical."""
    x = _extract_vec(p)
    n = len(x)
    f1 = 1 - math.exp(-4 * x[0]) * math.sin(6 * math.pi * x[0]) ** 6
    g = 1 + 9 * (sum(x[1:]) / (n - 1)) ** 0.25
    f2 = g * (1 - (f1 / g) ** 2)
    return {"f1": f1, "f2": f2}


# ---------------------------------------------------------------------------
# True Pareto fronts (for metric computation)
# ---------------------------------------------------------------------------


def zdt1_pareto_front(n_points: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def zdt2_pareto_front(n_points: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - f1**2
    return np.column_stack([f1, f2])


def zdt3_pareto_front(n_points: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 0.8518, n_points)
    f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
    # Filter to non-dominated points
    front = np.column_stack([f1, f2])
    mask = np.ones(len(front), dtype=bool)
    for i in range(len(front)):
        if mask[i]:
            mask[i + 1 :] &= ~np.all(front[i + 1 :] >= front[i], axis=1)
    return front[mask]


def zdt4_pareto_front(n_points: int = 500) -> np.ndarray:
    # Same as ZDT1 when g=1 (global optimum)
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def zdt6_pareto_front(n_points: int = 500) -> np.ndarray:
    # ZDT6 front: f1 in [~0.2807, 1], f2 = 1 - f1^2
    f1 = np.linspace(0.2807753191, 1, n_points)
    f2 = 1 - f1**2
    return np.column_stack([f1, f2])
