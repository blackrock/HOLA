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

"""pymoo-based multi-objective optimizer adapters (NSGA-II, MOEA/D)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from benchmarks.adapters.base import MultiObjectiveResult
from benchmarks.problems.registry import MultiObjectiveProblem


class _MOBenchmarkProblem(ElementwiseProblem):
    """Wraps a MultiObjectiveProblem for pymoo."""

    def __init__(self, problem: MultiObjectiveProblem) -> None:
        self._problem = problem
        self._param_names = list(problem.bounds.keys())
        xl = np.array([problem.bounds[k][0] for k in self._param_names])
        xu = np.array([problem.bounds[k][1] for k in self._param_names])
        super().__init__(
            n_var=problem.dimensionality,
            n_obj=problem.n_objectives,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        params = dict(zip(self._param_names, x, strict=True))
        result = self._problem.func(params)
        out["F"] = [result[name] for name in self._problem.objective_names]


class PymooNSGAIIAdapter:
    """NSGA-II via pymoo."""

    name = "NSGA-II (pymoo)"

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        pymoo_problem = _MOBenchmarkProblem(problem)
        algorithm = NSGA2()
        termination = get_termination("n_eval", budget)

        t0 = time.perf_counter()
        res = minimize(pymoo_problem, algorithm, termination, verbose=False, seed=seed)
        wall_time = time.perf_counter() - t0

        front = res.F if res.F is not None else np.empty((0, problem.n_objectives))
        return MultiObjectiveResult(pareto_front=front, wall_time_seconds=wall_time)


class PymooMOEADAdapter:
    """MOEA/D via pymoo."""

    name = "MOEA/D"

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        pymoo_problem = _MOBenchmarkProblem(problem)

        # Reference directions for decomposition
        n_partitions = max(3, min(20, budget // 10))
        ref_dirs = get_reference_directions(
            "das-dennis", problem.n_objectives, n_partitions=n_partitions
        )
        algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=min(15, len(ref_dirs)))
        termination = get_termination("n_eval", budget)

        t0 = time.perf_counter()
        res = minimize(pymoo_problem, algorithm, termination, verbose=False, seed=seed)
        wall_time = time.perf_counter() - t0

        front = res.F if res.F is not None else np.empty((0, problem.n_objectives))
        return MultiObjectiveResult(pareto_front=front, wall_time_seconds=wall_time)
