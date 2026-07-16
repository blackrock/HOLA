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

"""Optuna-based optimizer adapters (TPE for SO, NSGA-II for MO)."""

from __future__ import annotations

import time

import numpy as np
import optuna

from benchmarks.adapters.base import (
    MultiObjectiveResult,
    SingleObjectiveResult,
    assert_exact_evaluations,
)
from benchmarks.problems.registry import (
    MultiObjectiveProblem,
    SingleObjectiveProblem,
)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTPEAdapter:
    """Tree-structured Parzen Estimator via Optuna."""

    name = "TPE"

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "sampler": "TPESampler",
            "sampler_parameters": "documented defaults",
            "budget": budget,
        }

    def optimize(
        self,
        problem: SingleObjectiveProblem,
        budget: int,
        seed: int,
    ) -> SingleObjectiveResult:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler, direction="minimize")

        best_so_far = float("inf")
        trace: list[float] = []

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_so_far
            params = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in problem.bounds.items()}
            value = problem.func(params)
            best_so_far = min(best_so_far, value)
            trace.append(best_so_far)
            return value

        t0 = time.perf_counter()
        study.optimize(objective, n_trials=budget, show_progress_bar=False)
        wall_time = time.perf_counter() - t0
        n_evaluations = len(study.trials)
        assert_exact_evaluations(n_evaluations, budget, self.name)

        return SingleObjectiveResult(
            best_value=study.best_value,
            best_params=study.best_params,
            wall_time_seconds=wall_time,
            n_evaluations=n_evaluations,
            convergence_trace=trace,
        )


class OptunaNSGAIIAdapter:
    """NSGA-II via Optuna for multi-objective optimization."""

    name = "NSGA-II (Optuna)"

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "sampler": "NSGAIISampler",
            "sampler_parameters": "documented defaults",
            "budget": budget,
        }

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
        study = optuna.create_study(
            sampler=sampler,
            directions=["minimize"] * problem.n_objectives,
        )

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            params = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in problem.bounds.items()}
            result = problem.func(params)
            return tuple(result[name] for name in problem.objective_names)

        t0 = time.perf_counter()
        study.optimize(objective, n_trials=budget, show_progress_bar=False)
        wall_time = time.perf_counter() - t0
        n_evaluations = len(study.trials)
        assert_exact_evaluations(n_evaluations, budget, self.name)

        # Extract Pareto front from best trials
        selected_trials = []
        front_rows: list[list[int | float]] = []
        for trial in study.best_trials:
            values = trial.values
            if (
                values is not None
                and all(np.isfinite(value) for value in values)
                and not problem.is_infeasible_objectives(values)
            ):
                selected_trials.append(trial)
                front_rows.append(list(values))
        if selected_trials:
            front = np.array(front_rows)
            param_names = list(problem.bounds)
            decision_vectors = np.asarray(
                [[trial.params[name] for name in param_names] for trial in selected_trials],
                dtype=object,
            )
        else:
            front = np.empty((0, problem.n_objectives))
            decision_vectors = np.empty((0, problem.dimensionality), dtype=object)

        return MultiObjectiveResult(
            pareto_front=front,
            wall_time_seconds=wall_time,
            n_evaluations=n_evaluations,
            decision_vectors=decision_vectors,
        )
