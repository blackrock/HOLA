# Copyright 2021 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import multiprocess
from multiprocess import Pool  # pylint: disable=no-name-in-module
from pandas import DataFrame

from hola.algorithm import HOLA
from hola.objective import FuncName, ObjectiveConfig, ObjectivesSpec
from hola.params import ParamConfig, ParamName, ParamsSpec


def tune(
    func: Callable[..., dict[str, float]],
    params: ParamsSpec,
    objectives: ObjectivesSpec,
    num_runs: int = 100,
    n_jobs: int = 1,
    min_samples: int | None = None,
) -> Tuner:
    """
    :param func: multi-objective function to minimize, takes in float arguments and returns a dictionary
                 {objective_name: objective_value}
    :param params: search space for params
    :param objectives: Target-Priority-Limit scalarizer
    :param num_runs: number of evaluations of the multi-objective function
    :param n_jobs: number of processes (by default, equal to the number of CPU cores)
    :return: Tuner object
    """
    n_dim = len(params)
    if min_samples is None:
        min_samples = min(num_runs // 5, 10 * n_dim + 10)
    top_frac = 0.2
    n_components = safe_n_components(n_dim + 1, min_samples, top_frac)
    tuner = Tuner(
        HOLA(
            params,
            objectives,
            top_frac=top_frac,
            min_samples=min_samples,
            n_components=n_components,
            min_fit_samples=n_components,
            gmm_sampler="uniform",
            explore_sampler="sobol",
        )
    )
    tuner.tune(func, num_runs, n_jobs)
    return tuner


class Tuner:
    def __init__(self, hola: HOLA):
        self.hola = hola

    @property
    def params_config(self) -> dict[ParamName, ParamConfig]:
        return self.hola.params_config

    @property
    def objectives_config(self) -> dict[FuncName, ObjectiveConfig]:
        return self.hola.objectives_config

    def tune(
        self, func: Callable[..., dict[str, float]], num_runs: int = 100, n_jobs: int = -1
    ) -> list[tuple[dict[str, float], dict[ParamName, float]]]:
        if n_jobs == -1:
            n_jobs = multiprocess.cpu_count()
            print(f"Running hyperparameter tuning with {n_jobs} cpus")
        results: list[tuple[dict[str, float], dict[ParamName, float]]] = []
        if n_jobs == 1:
            # single process
            for _ in range(num_runs):
                params = self.hola.sample()
                result = func(**params)
                self.hola.add_run(result, params)
                results.append((result, params))
        else:
            with Pool(n_jobs) as pool:  # pylint: disable=not-callable
                while len(results) < num_runs:
                    args = [_PoolArgs(func, self.hola.sample()) for _ in range(n_jobs)]
                    res = pool.map(_func_helper, args)
                    for arg, res_ in zip(args, res):
                        self.hola.add_run(res_, arg.params)
                        results.append((res_, arg.params))
        return results

    def get_best_params(self) -> dict[ParamName, float]:
        return self.hola.get_best_params()

    def get_best_scores(self) -> dict[FuncName, float]:
        return self.hola.get_best_scores()

    def get_leaderboard(self) -> DataFrame:
        return self.hola.get_leaderboard()

    def save(self, file: Union[Path, str]) -> None:
        self.hola.save(file)

    def load(self, file: Union[Path, str]) -> None:
        self.hola.load(file)


@dataclass
class _PoolArgs:
    func: Callable[..., dict[str, float]]
    params: dict[str, float]


def _func_helper(pool_args: _PoolArgs) -> dict[str, float]:
    return pool_args.func(**pool_args.params)


def safe_n_components(proposed_n_components: int, min_samples: int, top_frac: float) -> int:
    max_components = int(min_samples * top_frac)
    if max_components == 0:
        return 1
    if max_components < proposed_n_components:
        return max_components
    return proposed_n_components
