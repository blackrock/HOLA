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
)
from benchmarks.problems.registry import (
    MultiObjectiveProblem,
    SingleObjectiveProblem,
)
from hola_opt import Minimize, Real, Space, Study


class HolaSingleObjectiveAdapter:
    """HOLA single-objective adapter with configurable strategy."""

    def __init__(self, strategy: str = "gmm") -> None:
        self.strategy = strategy
        label = "GMM" if strategy == "gmm" else strategy
        self.name = f"HOLA ({label})"

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
        )

        def wrapped(params: dict) -> dict:
            return {"value": problem.func(params)}

        t0 = time.perf_counter()
        study.run(wrapped, budget, n_workers=1)
        wall_time = time.perf_counter() - t0

        # Reconstruct convergence trace from trial history
        best_so_far = float("inf")
        trace: list[float] = []
        for trial in study.trials():
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
            convergence_trace=trace,
        )


class HolaMultiObjectiveAdapter:
    """HOLA multi-objective adapter.

    Each objective maps to its own priority group via distinct priority values.
    """

    def __init__(self, strategy: str = "gmm") -> None:
        self.strategy = strategy
        label = "GMM" if strategy == "gmm" else strategy
        self.name = f"HOLA MO ({label})"

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult:
        space_kwargs = {k: Real(lo, hi) for k, (lo, hi) in problem.bounds.items()}

        # Each objective gets a distinct priority to form separate groups
        objectives = []
        for i, obj_name in enumerate(problem.objective_names):
            ref = problem.reference_point[i]
            objectives.append(Minimize(obj_name, target=0.0, limit=ref, priority=float(i + 1)))

        study = Study(
            space=Space(**space_kwargs),
            objectives=objectives,
            strategy=self.strategy,
            seed=seed,
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
        if all_trials:
            raw_objectives = np.array(
                [[t.metrics[name] for name in problem.objective_names] for t in all_trials]
            )
            fronts = NonDominatedSorting().do(raw_objectives)
            pareto_array = raw_objectives[fronts[0]]
        else:
            pareto_array = np.empty((0, len(problem.objective_names)))

        return MultiObjectiveResult(
            pareto_front=pareto_array,
            wall_time_seconds=wall_time,
        )
