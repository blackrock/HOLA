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

"""Single-objective benchmark test functions.

Ported from https://github.com/blackrock/HOLA.
References: https://www.sfu.ca/~ssurjano/optimization.html
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_vec(p: dict[str, float]) -> list[float]:
    """Extract ordered vector from dict with keys x0, x1, ... or x, x1, x2, ..."""
    keys = sorted(k for k in p if k.startswith("x"))
    return [p[k] for k in keys]


# ---------------------------------------------------------------------------
# 1-D functions
# ---------------------------------------------------------------------------


def forrester(p: dict[str, float]) -> float:
    """Forrester et al. (2008). Domain: x in [0, 1]. Min: -6.0267."""
    x = p["x"]
    return (6 * x - 2) ** 2 * math.sin(12 * x - 4)


def gramacy_lee(p: dict[str, float]) -> float:
    """Gramacy & Lee (2012). Domain: x in [0.5, 2.5]. Min: ~-0.869."""
    x = p["x"]
    return math.sin(10 * math.pi * x) / (2 * x) + (x - 1) ** 4


# ---------------------------------------------------------------------------
# 2-D functions
# ---------------------------------------------------------------------------


def branin(p: dict[str, float]) -> float:
    """Branin-Hoo. Domain: x1 in [-5, 10], x2 in [0, 15]. Min: 0.397887."""
    x1, x2 = p["x1"], p["x2"]
    a, b, c = 1, 5.1 / (4 * math.pi**2), 5 / math.pi
    r, s, t = 6, 10, 1 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def bukin_6(p: dict[str, float]) -> float:
    """Bukin N.6. Domain: x1 in [-15, -5], x2 in [-3, 3]. Min: 0."""
    x1, x2 = p["x1"], p["x2"]
    return 100 * math.sqrt(abs(x2 - 0.01 * x1**2)) + 0.01 * abs(x1 + 10)


def cross_in_tray(p: dict[str, float]) -> float:
    """Cross-in-Tray. Domain: x1, x2 in [-10, 10]. Min: -2.06261."""
    x1, x2 = p["x1"], p["x2"]
    inner = abs(
        math.sin(x1) * math.sin(x2) * math.exp(abs(100 - math.sqrt(x1**2 + x2**2) / math.pi))
    )
    return -0.0001 * (inner + 1) ** 0.1


def drop_wave(p: dict[str, float]) -> float:
    """Drop-Wave. Domain: x1, x2 in [-5.12, 5.12]. Min: -1."""
    x1, x2 = p["x1"], p["x2"]
    sq_sum = x1**2 + x2**2
    return -(1 + math.cos(12 * math.sqrt(sq_sum))) / (0.5 * sq_sum + 2)


def egg_holder(p: dict[str, float]) -> float:
    """Egg Holder. Domain: x1, x2 in [-512, 512]. Min: -959.6407."""
    x1, x2 = p["x1"], p["x2"]
    t1 = (x2 + 47) * math.sin(math.sqrt(abs(x2 + x1 / 2 + 47)))
    t2 = x1 * math.sin(math.sqrt(abs(x1 - (x2 + 47))))
    return -t1 - t2


def holder_table(p: dict[str, float]) -> float:
    """Holder Table. Domain: x1, x2 in [-10, 10]. Min: -19.2085."""
    x1, x2 = p["x1"], p["x2"]
    return -abs(math.sin(x1) * math.cos(x2) * math.exp(abs(1 - math.sqrt(x1**2 + x2**2) / math.pi)))


def levy_13(p: dict[str, float]) -> float:
    """Levy N.13. Domain: x1, x2 in [-10, 10]. Min: 0."""
    x1, x2 = p["x1"], p["x2"]
    t1 = math.sin(3 * math.pi * x1) ** 2
    t2 = (x1 - 1) ** 2 * (1 + math.sin(3 * math.pi * x2) ** 2)
    t3 = (x2 - 1) ** 2 * (1 + math.sin(2 * math.pi * x1) ** 2)
    return t1 + t2 + t3


def schaffer_2(p: dict[str, float]) -> float:
    """Schaffer N.2. Domain: x1, x2 in [-100, 100]. Min: 0."""
    x1, x2 = p["x1"], p["x2"]
    num = math.sin(x1**2 - x2**2) ** 2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2)) ** 2
    return 0.5 + num / den


def schaffer_4(p: dict[str, float]) -> float:
    """Schaffer N.4. Domain: x1, x2 in [-100, 100]. Min: 0.292579."""
    x1, x2 = p["x1"], p["x2"]
    num = math.cos(math.sin(abs(x1**2 - x2**2))) ** 2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2)) ** 2
    return 0.5 + num / den


def shubert(p: dict[str, float]) -> float:
    """Shubert. Domain: x1, x2 in [-10, 10]. Min: -186.7309."""
    x1, x2 = p["x1"], p["x2"]
    s1 = sum(i * math.cos((i + 1) * x1 + i) for i in range(1, 6))
    s2 = sum(i * math.cos((i + 1) * x2 + i) for i in range(1, 6))
    return s1 * s2


def six_hump_camel(p: dict[str, float]) -> float:
    """Six-Hump Camel. Domain: x1 in [-3, 3], x2 in [-2, 2]. Min: -1.0316."""
    x1, x2 = p["x1"], p["x2"]
    t1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
    t2 = x1 * x2
    t3 = (-4 + 4 * x2**2) * x2**2
    return t1 + t2 + t3


# ---------------------------------------------------------------------------
# N-D functions (keys: x0, x1, ..., x_{n-1})
# ---------------------------------------------------------------------------


def ackley(p: dict[str, float]) -> float:
    """Ackley. Domain: xi in [-32.768, 32.768]. Min: 0."""
    x = _extract_vec(p)
    n = len(x)
    sum_sq = sum(xi**2 for xi in x)
    sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(sum_sq / n)) - math.exp(sum_cos / n) + 20 + math.e


def michalewicz(p: dict[str, float]) -> float:
    """Michalewicz. Domain: xi in [0, pi]. Min: depends on n."""
    x = _extract_vec(p)
    m = 10
    return -sum(
        math.sin(xi) * math.sin((i + 1) * xi**2 / math.pi) ** (2 * m) for i, xi in enumerate(x)
    )


def rastrigin(p: dict[str, float]) -> float:
    """Rastrigin. Domain: xi in [-5.12, 5.12]. Min: 0."""
    x = _extract_vec(p)
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock(p: dict[str, float]) -> float:
    """Rosenbrock. Domain: xi in [-5, 10]. Min: 0."""
    x = _extract_vec(p)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))


def schwefel(p: dict[str, float]) -> float:
    """Schwefel. Domain: xi in [-500, 500]. Min: 0."""
    x = _extract_vec(p)
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)
