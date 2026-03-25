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

Decision variables z_i in [0, 2i] for i = 1..n. We normalize internally
so the interface accepts x0..x_{n-1} in [0, 1].

We implement WFG1, WFG4, and WFG9 as specified in benchmark_design.md.
"""

from __future__ import annotations

import math

import numpy as np


def _extract_vec(p: dict[str, float]) -> np.ndarray:
    keys = sorted(k for k in p if k.startswith("x"))
    return np.array([p[k] for k in keys])


# ---------------------------------------------------------------------------
# WFG transformation functions
# ---------------------------------------------------------------------------


def _s_linear(y: float, a: float = 0.35) -> float:
    return abs(y - a) / abs(math.floor(a - y) + a)


def _b_flat(y: float, a: float, b: float, c: float) -> float:
    return (
        a
        + min(0.0, math.floor(y - b)) * a * (b - y) / b
        - min(0.0, math.floor(c - y)) * (1 - a) * (y - c) / (1 - c)
    )


def _b_poly(y: float, alpha: float) -> float:
    return max(0.0, y) ** alpha


def _s_multi(y: float, a: int, b: float, c: float) -> float:
    tmp = abs(y - c) / (2 * (math.floor(c - y) + c))
    return (1 + math.cos((4 * a + 2) * math.pi * (0.5 - tmp)) + 4 * b * tmp**2) / (1 + 4 * b)


def _s_decept(y: float, a: float = 0.35, b: float = 0.001, c: float = 0.05) -> float:
    q1 = math.floor(y - a + b) * (1 - c + (a - b) / b) / (a - b)
    q2 = math.floor(a + b - y) * (1 - c + (1 - a - b) / b) / (1 - a - b)
    return 1 + (abs(y - a) - b) * (q1 + q2 + 1 / b)


def _r_sum(y: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(y, w) / w.sum())


def _r_nonsep(y: np.ndarray, a: int) -> float:
    n = len(y)
    total = 0.0
    for j in range(n):
        total += y[j]
        for k in range(a - 1):
            total += abs(y[j] - y[(j + k + 1) % n])
    return total / (n / a * math.ceil(a / 2) * (1 + 2 * a - 2 * math.ceil(a / 2)))


# ---------------------------------------------------------------------------
# WFG shape functions
# ---------------------------------------------------------------------------


def _convex(x: np.ndarray, m: int, n_obj: int) -> float:
    result = 1.0
    for i in range(n_obj - m):
        result *= 1 - math.cos(x[i] * math.pi / 2)
    if m > 1:
        result *= 1 - math.sin(x[n_obj - m] * math.pi / 2)
    return result


def _concave(x: np.ndarray, m: int, n_obj: int) -> float:
    result = 1.0
    for i in range(n_obj - m):
        result *= math.sin(x[i] * math.pi / 2)
    if m > 1:
        result *= math.cos(x[n_obj - m] * math.pi / 2)
    return result


def _mixed(x: float, a: int = 5, alpha: float = 1.0) -> float:
    return (1 - x - math.cos(2 * a * math.pi * x + math.pi / 2) / (2 * a * math.pi)) ** alpha


def _disc(x: float, a: int = 5, alpha: float = 0.1, beta: float = 0.1) -> float:
    return 1 - x**alpha * math.cos(a * x**beta * math.pi) ** 2


# ---------------------------------------------------------------------------
# WFG problems
# ---------------------------------------------------------------------------


def wfg1(p: dict[str, float], n_obj: int = 2) -> dict[str, float]:
    """WFG1: mixed convex/concave, biased. Separable."""
    z = _extract_vec(p)
    n = len(z)
    k = n_obj - 1  # position parameters

    # Normalize: z_i in [0, 2i] -> y_i in [0, 1]
    y = z.copy()  # already in [0,1] from our interface

    # Transition 1: bias polynomial on distance params
    t1 = np.copy(y)
    for i in range(k, n):
        t1[i] = _b_poly(y[i], 0.02)

    # Transition 2: bias flat on distance params
    t2 = np.copy(t1)
    for i in range(k, n):
        t2[i] = _b_flat(t1[i], 0.8, 0.75, 0.85)

    # Transition 3: bias polynomial on all
    t3 = np.array([_b_poly(t2[i], 0.02) for i in range(n)])

    # Transition 4: reduction via weighted sum
    w = np.array([2.0 * (i + 1) for i in range(n)])
    t4 = np.empty(n_obj)
    for i in range(n_obj - 1):
        start = i * k // (n_obj - 1) if n_obj > 1 else 0
        end = (i + 1) * k // (n_obj - 1) if n_obj > 1 else k
        idx = slice(start, end)
        t4[i] = _r_sum(t3[idx], w[idx])
    t4[n_obj - 1] = _r_sum(t3[k:], w[k:])

    # Shape: convex + mixed
    x = np.empty(n_obj)
    for i in range(n_obj - 1):
        x[i] = max(t4[i], 1e-10) * (t4[i] - 0.5) + 0.5 if abs(t4[i] - 0.5) >= 1e-10 else t4[i]
    x[n_obj - 1] = t4[n_obj - 1]

    objs = {}
    d = 1
    s = [2.0 * (i + 1) for i in range(n_obj)]
    for m in range(1, n_obj):
        objs[f"f{m}"] = d * x[n_obj - 1] + s[m - 1] * _convex(x, m, n_obj)
    objs[f"f{n_obj}"] = d * x[n_obj - 1] + s[n_obj - 1] * _mixed(x[0], 5, 1.0)
    return objs


def wfg4(p: dict[str, float], n_obj: int = 2) -> dict[str, float]:
    """WFG4: multimodal. Separable."""
    z = _extract_vec(p)
    n = len(z)
    k = n_obj - 1
    y = z.copy()

    # Transition 1: multi-modal shift
    t1 = np.array([_s_multi(y[i], 30, 10, 0.35) for i in range(n)])

    # Transition 2: weighted sum reduction
    w = np.ones(n)
    t2 = np.empty(n_obj)
    for i in range(n_obj - 1):
        start = i * k // (n_obj - 1) if n_obj > 1 else 0
        end = (i + 1) * k // (n_obj - 1) if n_obj > 1 else k
        idx = slice(start, end)
        t2[i] = _r_sum(t1[idx], w[idx])
    t2[n_obj - 1] = _r_sum(t1[k:], w[k:])

    # Shape: concave
    x = np.empty(n_obj)
    for i in range(n_obj - 1):
        x[i] = max(t2[i], 1e-10) * (t2[i] - 0.5) + 0.5 if abs(t2[i] - 0.5) >= 1e-10 else t2[i]
    x[n_obj - 1] = t2[n_obj - 1]

    objs = {}
    d = 1
    s = [2.0 * (i + 1) for i in range(n_obj)]
    for m in range(1, n_obj + 1):
        objs[f"f{m}"] = d * x[n_obj - 1] + s[m - 1] * _concave(x, m, n_obj)
    return objs


def wfg9(p: dict[str, float], n_obj: int = 2) -> dict[str, float]:
    """WFG9: non-separable, deceptive."""
    z = _extract_vec(p)
    n = len(z)
    k = n_obj - 1
    y = z.copy()

    # Transition 1: dependent bias (each variable depends on remaining)
    t1 = np.empty(n)
    for i in range(n - 1):
        w = np.ones(n - i - 1)
        u = _r_sum(y[i + 1 :], w)
        t1[i] = _b_poly(y[i], 0.02 + 50 * (1 - u))  # param-dependent bias
    t1[n - 1] = y[n - 1]

    # Transition 2: shift deceptive on position, multi on distance
    t2 = np.empty(n)
    for i in range(k):
        t2[i] = _s_decept(t1[i], 0.35, 0.001, 0.05)
    for i in range(k, n):
        t2[i] = _s_multi(t1[i], 30, 95, 0.35)

    # Transition 3: non-separable reduction
    a = k // (n_obj - 1) if n_obj > 1 else 1
    t3 = np.empty(n_obj)
    for i in range(n_obj - 1):
        start = i * k // (n_obj - 1) if n_obj > 1 else 0
        end = (i + 1) * k // (n_obj - 1) if n_obj > 1 else k
        t3[i] = _r_nonsep(t2[start:end], a)
    t3[n_obj - 1] = _r_nonsep(t2[k:], n - k)

    # Shape: concave
    x = np.empty(n_obj)
    for i in range(n_obj - 1):
        x[i] = max(t3[i], 1e-10) * (t3[i] - 0.5) + 0.5 if abs(t3[i] - 0.5) >= 1e-10 else t3[i]
    x[n_obj - 1] = t3[n_obj - 1]

    objs = {}
    d = 1
    s = [2.0 * (i + 1) for i in range(n_obj)]
    for m in range(1, n_obj + 1):
        objs[f"f{m}"] = d * x[n_obj - 1] + s[m - 1] * _concave(x, m, n_obj)
    return objs
