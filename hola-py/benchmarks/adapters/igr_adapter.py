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

"""Iterative Grid Refinement (IGR) optimizer adapter.

Reimplemented from the upstream HOLA Python package.
"""

from __future__ import annotations

import itertools
import time
import warnings

from benchmarks.adapters.base import SingleObjectiveResult
from benchmarks.problems.registry import SingleObjectiveProblem


class IGRAdapter:
    """Iterative Grid Refinement.

    Starts with a grid search over the full domain, finds the best point,
    shrinks the search region around it, and repeats.
    """

    name = "IGR"

    def __init__(self, spacing: int = 4) -> None:
        self.spacing = spacing

    def optimize(
        self,
        problem: SingleObjectiveProblem,
        budget: int,
        seed: int,
    ) -> SingleObjectiveResult:
        n_dim = problem.dimensionality
        if n_dim >= 4:
            warnings.warn("IGR is not recommended for dimensionality >= 4", stacklevel=2)

        param_names = list(problem.bounds.keys())

        # Work in normalized [0, 1]^n space
        lower = [0.0] * n_dim
        upper = [1.0] * n_dim

        best_value = float("inf")
        best_params: dict[str, float] = {}
        trace: list[float] = []
        tried: set[tuple[float, ...]] = set()
        evals = 0

        t0 = time.perf_counter()

        while evals < budget:
            # Generate grid points in current hypercube
            lattices_per_dim = [
                [lo + (hi - lo) * i / self.spacing for i in range(self.spacing + 1)]
                for lo, hi in zip(lower, upper, strict=True)
            ]
            grid_points = itertools.product(*lattices_per_dim)

            best_in_gen_value = float("inf")
            best_in_gen_norm: tuple[float, ...] | None = None

            for norm_point in grid_points:
                if evals >= budget:
                    break
                if norm_point in tried:
                    continue
                tried.add(norm_point)

                # Denormalize to domain
                params = {}
                for j, name in enumerate(param_names):
                    lo, hi = problem.bounds[name]
                    params[name] = lo + norm_point[j] * (hi - lo)

                value = problem.func(params)
                evals += 1

                if value < best_value:
                    best_value = value
                    best_params = params
                if value < best_in_gen_value:
                    best_in_gen_value = value
                    best_in_gen_norm = norm_point

                trace.append(best_value)

            if best_in_gen_norm is None:
                break

            # Shrink hypercube around best point in this generation
            for j in range(n_dim):
                half_step = (upper[j] - lower[j]) / (2 * self.spacing)
                lower[j] = max(0.0, best_in_gen_norm[j] - half_step)
                upper[j] = min(1.0, best_in_gen_norm[j] + half_step)

        wall_time = time.perf_counter() - t0

        return SingleObjectiveResult(
            best_value=best_value,
            best_params=best_params,
            wall_time_seconds=wall_time,
            convergence_trace=trace,
        )
