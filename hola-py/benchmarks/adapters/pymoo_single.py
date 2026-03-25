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

"""pymoo-based single-objective optimizer adapters."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from benchmarks.adapters.base import SingleObjectiveResult
from benchmarks.problems.registry import SingleObjectiveProblem


class _BenchmarkProblem(ElementwiseProblem):
    """Wraps a SingleObjectiveProblem for pymoo."""

    def __init__(self, problem: SingleObjectiveProblem) -> None:
        self._problem = problem
        self._param_names = list(problem.bounds.keys())
        xl = np.array([problem.bounds[k][0] for k in self._param_names])
        xu = np.array([problem.bounds[k][1] for k in self._param_names])
        super().__init__(n_var=problem.dimensionality, n_obj=1, xl=xl, xu=xu)
        self.trace: list[float] = []
        self._best_so_far = float("inf")

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        params = dict(zip(self._param_names, x, strict=True))
        value = self._problem.func(params)
        out["F"] = [value]
        self._best_so_far = min(self._best_so_far, value)
        self.trace.append(self._best_so_far)


class PymooSingleAdapter:
    """Generic pymoo single-objective adapter."""

    def __init__(self, name: str, algorithm_factory: Any) -> None:
        self.name = name
        self._algorithm_factory = algorithm_factory

    def optimize(
        self,
        problem: SingleObjectiveProblem,
        budget: int,
        seed: int,
    ) -> SingleObjectiveResult:
        pymoo_problem = _BenchmarkProblem(problem)
        algorithm = self._algorithm_factory(budget, seed)
        termination = get_termination("n_eval", budget)

        t0 = time.perf_counter()
        res = minimize(pymoo_problem, algorithm, termination, verbose=False, seed=seed)
        wall_time = time.perf_counter() - t0

        best_params = dict(
            zip(
                list(problem.bounds.keys()),
                res.X if res.X is not None else [0.0] * problem.dimensionality,
                strict=True,
            )
        )

        return SingleObjectiveResult(
            best_value=float(res.F[0]) if res.F is not None else float("inf"),
            best_params=best_params,
            wall_time_seconds=wall_time,
            convergence_trace=pymoo_problem.trace,
        )


def ga_adapter() -> PymooSingleAdapter:
    return PymooSingleAdapter(
        name="GA",
        algorithm_factory=lambda budget, seed: GA(),
    )


def pso_adapter() -> PymooSingleAdapter:
    return PymooSingleAdapter(
        name="PSO",
        algorithm_factory=lambda budget, seed: PSO(),
    )


def nelder_mead_adapter() -> PymooSingleAdapter:
    return PymooSingleAdapter(
        name="Nelder-Mead",
        algorithm_factory=lambda budget, seed: NelderMead(),
    )


def hooke_jeeves_adapter() -> PymooSingleAdapter:
    return PymooSingleAdapter(
        name="Hooke-Jeeves",
        algorithm_factory=lambda budget, seed: PatternSearch(),
    )
