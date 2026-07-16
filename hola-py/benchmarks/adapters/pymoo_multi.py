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

"""pymoo-based multi-objective optimizer adapters (NSGA-II, MOEA/D).

NSGA-II keeps its documented population of 100 when the budget permits and
uses only budget-compatible offspring batches. MOEA/D uses up to 100
energy-spaced reference directions, limited by the budget. Both are driven by
the exact-budget ask/tell loop so objective-call counts cannot overshoot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from benchmarks.adapters.base import MultiObjectiveResult, assert_exact_evaluations
from benchmarks.adapters.pymoo_common import population_and_offspring_sizes, run_exact_budget
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
        self.objective_values: list[list[float]] = []
        self.decision_values: list[list[float]] = []
        self.n_evaluations = 0

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        params = dict(zip(self._param_names, x, strict=True))
        result = self._problem.func(params)
        values = [float(result[name]) for name in self._problem.objective_names]
        out["F"] = values
        self.objective_values.append(values)
        self.decision_values.append([float(value) for value in x])
        self.n_evaluations += 1

    def pareto_result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return non-dominated objectives and their corresponding decisions."""
        if not self.objective_values:
            return (
                np.empty((0, self._problem.n_objectives)),
                np.empty((0, self._problem.dimensionality)),
            )
        values = np.asarray(self.objective_values, dtype=float)
        decisions = np.asarray(self.decision_values, dtype=float)
        valid = np.all(np.isfinite(values), axis=1)
        if self._problem.infeasible_sentinel is not None:
            sentinel = np.asarray(self._problem.infeasible_sentinel, dtype=float)
            valid &= ~np.all(values == sentinel, axis=1)
        values = values[valid]
        decisions = decisions[valid]
        if len(values) == 0:
            return (
                np.empty((0, self._problem.n_objectives)),
                np.empty((0, self._problem.dimensionality)),
            )
        first_front = NonDominatedSorting().do(values, only_non_dominated_front=True)
        return values[first_front], decisions[first_front]


class PymooNSGAIIAdapter:
    """NSGA-II via pymoo."""

    name = "NSGA-II (pymoo)"

    def configuration(self, budget: int) -> dict[str, object]:
        population, offspring = population_and_offspring_sizes(budget, 100)
        return {
            "adapter": type(self).__name__,
            "algorithm": "NSGA2",
            "population_size": population,
            "offspring_size": offspring,
            "operators": "pymoo documented defaults",
            "execution": "exact-budget ask/tell",
        }

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        pymoo_problem = _MOBenchmarkProblem(problem)
        population, offspring = population_and_offspring_sizes(budget, 100)
        algorithm = NSGA2(pop_size=population, n_offsprings=offspring)
        wall_time = run_exact_budget(pymoo_problem, algorithm, budget, seed)
        assert_exact_evaluations(pymoo_problem.n_evaluations, budget, self.name)
        front, decision_vectors = pymoo_problem.pareto_result()

        return MultiObjectiveResult(
            pareto_front=front,
            wall_time_seconds=wall_time,
            n_evaluations=pymoo_problem.n_evaluations,
            decision_vectors=decision_vectors,
        )


class PymooMOEADAdapter:
    """MOEA/D via pymoo."""

    name = "MOEA/D"

    def configuration(self, budget: int) -> dict[str, object]:
        if budget < 2:
            raise ValueError("MOEA/D requires an evaluation budget of at least 2")
        population = min(100, budget)
        return {
            "adapter": type(self).__name__,
            "algorithm": "MOEAD",
            "population_size": population,
            "reference_directions": "energy",
            "n_neighbors": min(15, population),
            "operators": "pymoo documented defaults",
            "execution": "exact-budget ask/tell",
        }

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        pymoo_problem = _MOBenchmarkProblem(problem)

        if budget < 2:
            raise ValueError("MOEA/D requires an evaluation budget of at least 2")

        # Use exactly the requested number of energy-spaced directions instead
        # of Das-Dennis partitions, whose combinatorial population can exceed
        # the complete evaluation budget in many-objective problems.
        population = min(100, budget)
        ref_dirs = get_reference_directions("energy", problem.n_objectives, population, seed=seed)
        algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=min(15, len(ref_dirs)))
        wall_time = run_exact_budget(pymoo_problem, algorithm, budget, seed)
        assert_exact_evaluations(pymoo_problem.n_evaluations, budget, self.name)
        front, decision_vectors = pymoo_problem.pareto_result()

        return MultiObjectiveResult(
            pareto_front=front,
            wall_time_seconds=wall_time,
            n_evaluations=pymoo_problem.n_evaluations,
            decision_vectors=decision_vectors,
        )
