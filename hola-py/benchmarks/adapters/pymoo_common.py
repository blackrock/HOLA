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

"""Exact-evaluation-budget execution helpers for pymoo adapters.

pymoo checks its evaluation termination between algorithm steps. Population
generations and compound local-search steps can therefore overshoot an
``n_eval`` termination. Benchmark adapters instead drive pymoo's ask/tell API
directly and stop after exactly the declared number of objective calls.

Population adapters must use a task-independent budget-compatible batch size:
the initial population may not exceed the budget, and every later batch must
fit in the remaining budget. If an adapter violates that rule, the run fails
loudly instead of evaluating a partial generation or silently overshooting.
Loop-wise algorithms such as local search and MOEA/D naturally yield one point
at a time after initialization.
"""

from __future__ import annotations

import time
from typing import Any

from pymoo.core.individual import Individual
from pymoo.core.termination import NoTermination


def infill_size(infill: Any) -> int:
    """Return one for an Individual and len() for a Population."""
    return 1 if isinstance(infill, Individual) else len(infill)


def unevaluated_size(infill: Any, evaluate_values_of: list[str]) -> int:
    """Count candidates for which pymoo will make a new objective call."""
    individuals = [infill] if isinstance(infill, Individual) else infill
    return sum(
        not all(value in individual.evaluated for value in evaluate_values_of)
        for individual in individuals
    )


def run_exact_budget(problem: Any, algorithm: Any, budget: int, seed: int) -> float:
    """Drive a pymoo algorithm for exactly ``budget`` objective evaluations."""
    if budget <= 0:
        raise ValueError(f"evaluation budget must be positive, got {budget}")

    algorithm.setup(
        problem,
        seed=seed,
        termination=NoTermination(),
        verbose=False,
    )

    t0 = time.perf_counter()
    steps_without_evaluation = 0
    while algorithm.evaluator.n_eval < budget:
        infill = algorithm.ask()
        if infill is None or infill_size(infill) == 0:
            raise RuntimeError(
                f"{type(algorithm).__name__} produced no candidates before reaching "
                f"the {budget}-evaluation budget"
            )

        batch_size = unevaluated_size(infill, algorithm.evaluator.evaluate_values_of)
        remaining = budget - algorithm.evaluator.n_eval
        if batch_size > remaining:
            raise RuntimeError(
                f"{type(algorithm).__name__} proposed a batch of {batch_size} with only "
                f"{remaining} evaluations remaining; configure a budget-compatible batch size"
            )

        before = algorithm.evaluator.n_eval
        algorithm.evaluator.eval(problem, infill, algorithm=algorithm)
        used = algorithm.evaluator.n_eval - before
        if used != batch_size:
            raise RuntimeError(
                f"{type(algorithm).__name__} evaluated {used} of {batch_size} proposed candidates"
            )
        algorithm.tell(infills=infill)
        if used == 0:
            steps_without_evaluation += 1
            if steps_without_evaluation > 1000:
                raise RuntimeError(
                    f"{type(algorithm).__name__} made no objective calls for 1000 consecutive steps"
                )
        else:
            steps_without_evaluation = 0

    return time.perf_counter() - t0


def population_and_offspring_sizes(budget: int, default_population: int) -> tuple[int, int]:
    """Keep the default population when possible and make later batches fit.

    For budgets below the documented default, the initial population is reduced
    to the budget. Otherwise the default is retained. The offspring batch is
    the greatest common divisor of the initial population and remaining budget,
    which preserves full generations for the approved benchmark budgets and
    gives an exact, deterministic fallback for other budgets.
    """
    if budget <= 0:
        raise ValueError(f"evaluation budget must be positive, got {budget}")
    population = min(default_population, budget)
    remaining = budget - population
    if remaining == 0:
        return population, population

    import math

    return population, math.gcd(population, remaining)


def divisor_population_size(budget: int, default_population: int, optimizer: str) -> int:
    """Choose the largest default-compatible population dividing the budget."""
    if budget <= 0:
        raise ValueError(f"evaluation budget must be positive, got {budget}")
    for population in range(min(default_population, budget), 1, -1):
        if budget % population == 0:
            return population
    raise ValueError(
        f"{optimizer} requires a budget with a population-size divisor between 2 and "
        f"{min(default_population, budget)}; got {budget}"
    )
