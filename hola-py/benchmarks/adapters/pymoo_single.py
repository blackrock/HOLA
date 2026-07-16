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

"""pymoo-based single-objective optimizer adapters.

Primary population methods keep their documented default population when the
budget allows it and reduce only what is needed to honor an exact objective-call
budget. Hooke--Jeeves keeps pymoo's defaults except that its initial design is
capped by the total budget. Execution is driven by the shared exact-budget
ask/tell loop rather than pymoo's generation-boundary ``n_eval`` termination.

The Nelder--Mead adapter is retained only for explicit diagnostic runs. Its
path-dependent compound proposals do not fit every approved primary horizon;
the exact-budget driver fails loudly instead of splitting, padding, or
overshooting such a proposal.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem

from benchmarks.adapters.base import SingleObjectiveResult, assert_exact_evaluations
from benchmarks.adapters.pymoo_common import (
    divisor_population_size,
    population_and_offspring_sizes,
    run_exact_budget,
)
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
        self.best_value = float("inf")
        self.best_params: dict[str, float] = {}
        self.n_evaluations = 0

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        params = {name: float(value) for name, value in zip(self._param_names, x, strict=True)}
        value = self._problem.func(params)
        out["F"] = [value]
        self.n_evaluations += 1
        if value < self.best_value:
            self.best_value = value
            self.best_params = params
        self.trace.append(self.best_value)


PymooSingleKind = Literal["ga", "pso", "nelder_mead", "hooke_jeeves"]

_DISPLAY_NAMES: dict[PymooSingleKind, str] = {
    "ga": "GA",
    "pso": "PSO",
    "nelder_mead": "Nelder-Mead",
    "hooke_jeeves": "Hooke-Jeeves",
}


def _settings(kind: PymooSingleKind, budget: int) -> dict[str, int | str]:
    """Resolve the fixed budget-compatible settings for one adapter kind."""
    if kind == "ga":
        population, offspring = population_and_offspring_sizes(budget, 100)
        return {
            "population_size": population,
            "offspring_size": offspring,
            "operators": "pymoo documented defaults",
        }
    if kind == "pso":
        return {
            "population_size": divisor_population_size(budget, 25, "PSO"),
            "parameters": "pymoo documented defaults",
        }
    return {
        "initial_sample_points": min(20, budget),
        "parameters": "pymoo documented defaults",
    }


def _algorithm(kind: PymooSingleKind, budget: int) -> Any:
    """Construct an algorithm without storing unpicklable local callables."""
    settings = _settings(kind, budget)
    if kind == "ga":
        return GA(
            pop_size=settings["population_size"],
            n_offsprings=settings["offspring_size"],
        )
    if kind == "pso":
        return PSO(pop_size=settings["population_size"])
    if kind == "nelder_mead":
        return NelderMead(n_sample_points=settings["initial_sample_points"])
    return PatternSearch(n_sample_points=settings["initial_sample_points"])


class PymooSingleAdapter:
    """Generic pymoo single-objective adapter."""

    def __init__(self, kind: PymooSingleKind) -> None:
        self.kind = kind
        self.name = _DISPLAY_NAMES[kind]

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "algorithm": self.name,
            **_settings(self.kind, budget),
            "execution": "exact-budget ask/tell",
            "protocol_role": (
                "non-primary diagnostic" if self.kind == "nelder_mead" else "primary"
            ),
        }

    def optimize(
        self,
        problem: SingleObjectiveProblem,
        budget: int,
        seed: int,
    ) -> SingleObjectiveResult:
        pymoo_problem = _BenchmarkProblem(problem)
        algorithm = _algorithm(self.kind, budget)
        wall_time = run_exact_budget(pymoo_problem, algorithm, budget, seed)
        assert_exact_evaluations(pymoo_problem.n_evaluations, budget, self.name)

        return SingleObjectiveResult(
            best_value=pymoo_problem.best_value,
            best_params=pymoo_problem.best_params,
            wall_time_seconds=wall_time,
            n_evaluations=pymoo_problem.n_evaluations,
            convergence_trace=pymoo_problem.trace,
        )


def ga_adapter() -> PymooSingleAdapter:
    return PymooSingleAdapter("ga")


def pso_adapter() -> PymooSingleAdapter:
    return PymooSingleAdapter("pso")


def nelder_mead_adapter() -> PymooSingleAdapter:
    """Return the non-primary diagnostic Nelder--Mead adapter."""
    return PymooSingleAdapter("nelder_mead")


def hooke_jeeves_adapter() -> PymooSingleAdapter:
    return PymooSingleAdapter("hooke_jeeves")
