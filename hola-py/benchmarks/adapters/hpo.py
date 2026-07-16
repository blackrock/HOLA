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

"""Native mixed-space adapters for the dedicated practical HPO campaign."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import optuna

from benchmarks.adapters.base import HpoOptimizationResult, assert_exact_evaluations
from benchmarks.problems.hpo import (
    CategoricalParameter,
    HpoProblem,
    IntegerParameter,
    RealParameter,
)
from hola_opt import Categorical, Integer, Maximize, Real, Space, Study

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_hola_parameter_map(
    problem: HpoProblem,
) -> dict[str, Integer | Real | Categorical]:
    """Translate explicit benchmark specs to HOLA's native parameter types."""
    translated: dict[str, Integer | Real | Categorical] = {}
    for name, specification in problem.parameters.items():
        if isinstance(specification, IntegerParameter):
            translated[name] = Integer(specification.minimum, specification.maximum)
        elif isinstance(specification, RealParameter):
            translated[name] = Real(
                specification.minimum,
                specification.maximum,
                scale=specification.scale,
            )
        elif isinstance(specification, CategoricalParameter):
            translated[name] = Categorical(list(specification.choices))
        else:  # pragma: no cover - closed union, retained for defensive runtime checks
            raise TypeError(f"unsupported HPO parameter specification for {name!r}")
    return translated


def suggest_optuna_params(problem: HpoProblem, trial: optuna.Trial) -> dict[str, Any]:
    """Translate the same explicit specs to equivalent native Optuna calls."""
    params: dict[str, Any] = {}
    for name, specification in problem.parameters.items():
        if isinstance(specification, IntegerParameter):
            params[name] = trial.suggest_int(name, specification.minimum, specification.maximum)
        elif isinstance(specification, RealParameter):
            params[name] = trial.suggest_float(
                name,
                specification.minimum,
                specification.maximum,
                log=specification.scale == "log10",
            )
        elif isinstance(specification, CategoricalParameter):
            params[name] = trial.suggest_categorical(name, list(specification.choices))
        else:  # pragma: no cover - closed union, retained for defensive runtime checks
            raise TypeError(f"unsupported HPO parameter specification for {name!r}")
    return params


class HolaHpoAdapter:
    """HOLA validation-only adapter using native mixed parameter types."""

    def __init__(self, strategy: str) -> None:
        if strategy not in {"random", "sobol", "gmm"}:
            raise ValueError(f"unsupported HOLA HPO strategy {strategy!r}")
        self.strategy = strategy
        label = "GMM" if strategy == "gmm" else strategy
        self.name = f"HOLA HPO ({label})"

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "strategy": self.strategy,
            "max_trials": budget,
            "n_workers": 1,
            "objective": "maximize fixed-split validation R2",
        }

    def optimize(
        self,
        problem: HpoProblem,
        evaluate_validation: Callable[[dict[str, Any]], float],
        budget: int,
        seed: int,
    ) -> HpoOptimizationResult:
        study = Study(
            space=Space(**build_hola_parameter_map(problem)),
            objectives=[Maximize(problem.objective_name)],
            strategy=self.strategy,
            seed=seed,
            max_trials=budget,
        )

        def objective(params: dict[str, Any]) -> dict[str, float]:
            return {problem.objective_name: evaluate_validation(params)}

        started = time.perf_counter()
        study.run(objective, budget, n_workers=1)
        wall_time = time.perf_counter() - started
        trials = study.trials(sorted_by="index", include_infeasible=True)
        n_evaluations = len(trials)
        assert_exact_evaluations(n_evaluations, budget, self.name)
        validation_trace = [float(trial.metrics[problem.objective_name]) for trial in trials]
        best = max(trials, key=lambda trial: float(trial.metrics[problem.objective_name]))
        return HpoOptimizationResult(
            best_validation_value=float(best.metrics[problem.objective_name]),
            best_params=dict(best.params),
            wall_time_seconds=wall_time,
            n_evaluations=n_evaluations,
            validation_trace=validation_trace,
        )


class OptunaTpeHpoAdapter:
    """Optuna TPE validation-only competitor on the equivalent mixed space."""

    name = "Optuna HPO (TPE)"

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "sampler": "TPESampler",
            "sampler_parameters": "documented defaults except paired seed",
            "budget": budget,
            "objective": "maximize fixed-split validation R2",
        }

    def optimize(
        self,
        problem: HpoProblem,
        evaluate_validation: Callable[[dict[str, Any]], float],
        budget: int,
        seed: int,
    ) -> HpoOptimizationResult:
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=seed),
            direction="maximize",
        )
        validation_trace: list[float] = []

        def objective(trial: optuna.Trial) -> float:
            value = evaluate_validation(suggest_optuna_params(problem, trial))
            validation_trace.append(value)
            return value

        started = time.perf_counter()
        study.optimize(objective, n_trials=budget, show_progress_bar=False)
        wall_time = time.perf_counter() - started
        n_evaluations = len(study.trials)
        assert_exact_evaluations(n_evaluations, budget, self.name)
        return HpoOptimizationResult(
            best_validation_value=float(study.best_value),
            best_params=dict(study.best_params),
            wall_time_seconds=wall_time,
            n_evaluations=n_evaluations,
            validation_trace=validation_trace,
        )
