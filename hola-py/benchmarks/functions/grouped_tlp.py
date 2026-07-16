# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analytically checkable grouped-TLP benchmark functions."""

from __future__ import annotations

import numpy as np


def _extract_vec(params: dict[str, float]) -> np.ndarray:
    keys = sorted((key for key in params if key.startswith("x")), key=lambda key: int(key[1:]))
    return np.asarray([params[key] for key in keys], dtype=float)


def synthetic_grouped_tlp(params: dict[str, float]) -> dict[str, float]:
    """Return four raw objectives with an exact grouped-TLP Pareto curve.

    ``x0`` controls the trade-off and the remaining coordinates contribute a
    non-negative nuisance penalty to every raw objective. With the registered
    targets and limits, the exact Pareto set has zero nuisance penalty and
    ``x0`` in ``[0.2, 0.8]``. The outer target plateaus are dominated, while
    parts of the search box genuinely violate a TLP limit.
    """
    x = _extract_vec(params)
    if x.shape != (5,):
        raise ValueError(f"synthetic grouped TLP expects 5 variables; got {len(x)}")

    tradeoff = x[0]
    complement = 1.0 - tradeoff
    nuisance = 0.05 * float(np.sum(x[1:] ** 2))
    return {
        "f1": float(tradeoff + nuisance),
        "f2": float(tradeoff**2 + nuisance),
        "f3": float(complement + nuisance),
        "f4": float(complement**2 + nuisance),
    }


def synthetic_grouped_tlp_pareto_front(n_points: int = 1001) -> np.ndarray:
    """Return a deterministic sampling of the exact two-group Pareto curve.

    Along the Pareto set, ``x0`` ranges from ``1/5`` to ``4/5`` and the
    nuisance coordinates are zero. The expressions below apply the registered
    target-limit normalizations and unequal within-group priorities directly.
    """
    if n_points < 2:
        raise ValueError("the grouped TLP reference front needs at least two points")
    linear_target = 0.2
    linear_limit = 0.9
    quadratic_target = linear_target**2
    quadratic_limit = linear_limit**2
    x = np.linspace(linear_target, 1.0 - linear_target, n_points)
    complement = 1.0 - x
    group_a = (x - linear_target) / (linear_limit - linear_target) + 2.0 * (
        x**2 - quadratic_target
    ) / (quadratic_limit - quadratic_target)
    group_b = 2.0 * (complement - linear_target) / (linear_limit - linear_target) + (
        complement**2 - quadratic_target
    ) / (quadratic_limit - quadratic_target)
    # Decimal endpoints can otherwise produce negative roundoff on the order
    # of 1e-16 even though a TLP cost is non-negative by construction.
    group_a = np.maximum(group_a, 0.0)
    group_b = np.maximum(group_b, 0.0)
    return np.column_stack((group_a, group_b))
