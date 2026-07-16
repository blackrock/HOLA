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

"""HOLA optimizer adapters using the Study API."""

from __future__ import annotations

import math
import time

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from benchmarks.adapters.base import (
    MultiObjectiveResult,
    SingleObjectiveResult,
    assert_exact_evaluations,
)
from benchmarks.problems.registry import (
    GroupedTlpProblem,
    MultiObjectiveProblem,
    SingleObjectiveProblem,
)
from hola_opt import Maximize, Minimize, Real, Space, Study


class HolaSingleObjectiveAdapter:
    """HOLA single-objective adapter with configurable strategy."""

    def __init__(self, strategy: str = "gmm") -> None:
        self.strategy = strategy
        label = "GMM" if strategy == "gmm" else strategy
        self.name = f"HOLA ({label})"

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "strategy": self.strategy,
            "max_trials": budget,
            "n_workers": 1,
            "objective": "raw minimization",
        }

    def optimize(
        self,
        problem: SingleObjectiveProblem,
        budget: int,
        seed: int,
    ) -> SingleObjectiveResult:
        space_kwargs = {k: Real(lo, hi) for k, (lo, hi) in problem.bounds.items()}
        study = Study(
            space=Space(**space_kwargs),
            objectives=[Minimize("value")],
            strategy=self.strategy,
            seed=seed,
            max_trials=budget,
        )

        def wrapped(params: dict) -> dict:
            return {"value": problem.func(params)}

        t0 = time.perf_counter()
        study.run(wrapped, budget, n_workers=1)
        wall_time = time.perf_counter() - t0

        # Reconstruct convergence trace from trial history
        best_so_far = float("inf")
        trace: list[float] = []
        trials = study.trials()
        n_evaluations = len(trials)
        assert_exact_evaluations(n_evaluations, budget, self.name)
        for trial in trials:
            score = trial.score_vector.get("value", float("inf"))
            if math.isfinite(score):
                best_so_far = min(best_so_far, score)
            trace.append(best_so_far)

        top = study.top_k(1)
        best = top[0] if top else None
        return SingleObjectiveResult(
            best_value=best.score_vector["value"] if best else best_so_far,
            best_params=best.params if best else {},
            wall_time_seconds=wall_time,
            n_evaluations=n_evaluations,
            convergence_trace=trace,
        )


class HolaMultiObjectiveAdapter:
    """HOLA multi-objective adapter.

    Each objective is its own priority group (the group defaults to the field
    name), so all objectives are Pareto-ranked equally. Priorities are kept at
    1.0 to avoid weighting any objective more heavily than the others, matching
    the symmetric configuration used by the optuna/pymoo competitors.
    """

    def __init__(self, strategy: str = "gmm") -> None:
        self.strategy = strategy
        label = "GMM" if strategy == "gmm" else strategy
        self.name = f"HOLA MO ({label})"

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "strategy": self.strategy,
            "max_trials": budget,
            "n_workers": 1,
            "objectives": "unrestricted raw minimization, one group per field",
        }

    @staticmethod
    def _build_objectives(problem: MultiObjectiveProblem) -> list[Minimize | Maximize]:
        # Each objective is its own group (group defaults to the field name).
        # Leave target and limit unset so HOLA receives the same unrestricted
        # raw minimization objectives as every other adapter. Hypervolume
        # reference points are reporting geometry, not feasibility limits.
        objectives: list[Minimize | Maximize] = [
            Minimize(obj_name) for obj_name in problem.objective_names
        ]
        return objectives

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        space_kwargs = {k: Real(lo, hi) for k, (lo, hi) in problem.bounds.items()}

        objectives = self._build_objectives(problem)

        study = Study(
            space=Space(**space_kwargs),
            objectives=objectives,
            strategy=self.strategy,
            seed=seed,
            max_trials=budget,
        )

        t0 = time.perf_counter()
        study.run(problem.func, budget, n_workers=1)
        wall_time = time.perf_counter() - t0

        # Extract raw objective values from all completed trials and compute
        # the non-dominated front externally.  Using trial.metrics (raw values)
        # instead of trial.scores (TLP-transformed) ensures the Pareto front
        # and downstream metrics (HV, IGD) are computed on the same scale as
        # the external optimizers.
        all_trials = study.trials(sorted_by="index", include_infeasible=True)
        n_evaluations = len(all_trials)
        assert_exact_evaluations(n_evaluations, budget, self.name)
        if all_trials:
            raw_objectives = np.array(
                [[t.metrics[name] for name in problem.objective_names] for t in all_trials]
            )
            fronts = NonDominatedSorting().do(raw_objectives)
            first_front = fronts[0]
            pareto_array = raw_objectives[first_front]
            param_names = list(problem.bounds)
            decision_vectors = np.asarray(
                [[all_trials[index].params[name] for name in param_names] for index in first_front],
                dtype=object,
            )
        else:
            pareto_array = np.empty((0, len(problem.objective_names)))
            decision_vectors = np.empty((0, problem.dimensionality), dtype=object)

        return MultiObjectiveResult(
            pareto_front=pareto_array,
            wall_time_seconds=wall_time,
            n_evaluations=n_evaluations,
            decision_vectors=decision_vectors,
        )


class HolaGroupedTlpAdapter:
    """HOLA adapter that applies raw TLP objectives inside the study engine."""

    def __init__(self, grouped_problem: GroupedTlpProblem, strategy: str = "gmm") -> None:
        self.grouped_problem = grouped_problem
        self.strategy = strategy
        label = "GMM" if strategy == "gmm" else strategy
        self.name = f"HOLA grouped TLP ({label})"

    def configuration(self, budget: int) -> dict[str, object]:
        return {
            "adapter": type(self).__name__,
            "strategy": self.strategy,
            "problem": self.grouped_problem.name,
            "max_trials": budget,
            "n_workers": 1,
            "group_order": list(self.grouped_problem.group_names),
            "objectives": [
                {
                    "field": objective.field,
                    "sense": objective.sense,
                    "target": objective.target,
                    "limit": objective.limit,
                    "priority": objective.priority,
                    "group": objective.group,
                }
                for objective in self.grouped_problem.objectives
            ],
        }

    @staticmethod
    def _build_objectives(problem: GroupedTlpProblem) -> list[Minimize | Maximize]:
        objectives: list[Minimize | Maximize] = []
        for objective in problem.objectives:
            objective_type = Minimize if objective.sense == "minimize" else Maximize
            objectives.append(
                objective_type(
                    objective.field,
                    target=objective.target,
                    limit=objective.limit,
                    priority=objective.priority,
                    group=objective.group,
                )
            )
        return objectives

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        grouped = self.grouped_problem
        if (
            problem.name != grouped.name
            or problem.objective_names != grouped.group_names
            or problem.bounds != grouped.bounds
        ):
            raise ValueError(f"{self.name} is configured for {grouped.name}, not {problem.name}")

        study = Study(
            space=Space(
                **{name: Real(lower, upper) for name, (lower, upper) in grouped.bounds.items()}
            ),
            objectives=self._build_objectives(grouped),
            strategy=self.strategy,
            seed=seed,
            max_trials=budget,
        )

        started = time.perf_counter()
        study.run(grouped.raw_func, budget, n_workers=1)
        wall_time = time.perf_counter() - started

        trials = study.trials(sorted_by="index", include_infeasible=True)
        n_evaluations = len(trials)
        assert_exact_evaluations(n_evaluations, budget, self.name)
        if not trials:
            return MultiObjectiveResult(
                pareto_front=np.empty((0, grouped.n_groups)),
                decision_vectors=np.empty((0, grouped.dimensionality)),
                wall_time_seconds=wall_time,
                n_evaluations=0,
            )

        group_costs = np.asarray(
            [
                [float(trial.score_vector[group]) for group in grouped.group_names]
                for trial in trials
            ],
            dtype=float,
        )
        externally_transformed = np.asarray(
            [
                [grouped.group_costs(trial.metrics)[group] for group in grouped.group_names]
                for trial in trials
            ],
            dtype=float,
        )
        if not np.allclose(group_costs, externally_transformed, rtol=1e-12, atol=1e-12):
            raise RuntimeError(
                "HOLA's internal grouped TLP scores differ from the shared transform"
            )

        feasible_indices = np.flatnonzero(np.all(np.isfinite(group_costs), axis=1))
        if len(feasible_indices) == 0:
            return MultiObjectiveResult(
                pareto_front=np.empty((0, grouped.n_groups)),
                decision_vectors=np.empty((0, grouped.dimensionality)),
                wall_time_seconds=wall_time,
                n_evaluations=n_evaluations,
            )

        local_front = NonDominatedSorting().do(
            group_costs[feasible_indices],
            only_non_dominated_front=True,
        )
        first_front = feasible_indices[local_front]
        parameter_names = list(grouped.bounds)
        decision_vectors = np.asarray(
            [[trials[index].params[name] for name in parameter_names] for index in first_front],
            dtype=float,
        )
        return MultiObjectiveResult(
            pareto_front=group_costs[first_front],
            decision_vectors=decision_vectors,
            wall_time_seconds=wall_time,
            n_evaluations=n_evaluations,
        )
