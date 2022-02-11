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

from typing import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from hola.algorithm import HOLA
from hola.objective import FuncName, ObjectiveConfig, ObjectivesSpec
from hola.params import ParamConfig, ParamsSpec
from hola.tune import Tuner, tune

PARAMS_CONFIG: list[ParamsSpec] = [
    {"x_1": ParamConfig(min=-2.0, max=2.0), "x_2": ParamConfig(min=-1.0, max=3.0)},
    {"x_1": {"min": -2.0, "max": 2.0}, "x_2": {"min": -1.0, "max": 3.0}},
]
OBJECTIVE_CONFIG: list[ObjectivesSpec] = [
    {"f": ObjectiveConfig(target=10, limit=10)},
    {"f": {"target": 10.0, "limit": 10.0}},
]
SEED_0_SCORE = {"f": -332.84684342589117}


def rosenbrock(xs: Sequence[float] | npt.NDArray[np.floating]) -> float:
    x, y = xs
    a = 1.0 - x
    b = y - x * x
    return a * a + b * b * 100.0


def hyper_rose(x_1: float, x_2: float) -> dict[FuncName, float]:
    return {"f": -rosenbrock([x_1, x_2])}


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("param_config", PARAMS_CONFIG)
@pytest.mark.parametrize("objective_config", OBJECTIVE_CONFIG)
def test_tuner(n_jobs: int, param_config: ParamsSpec, objective_config: ObjectivesSpec) -> None:
    np.random.seed(0)
    tuner = Tuner(
        HOLA(
            params_config=param_config,
            objectives_config=objective_config,
            min_samples=30,
            gmm_sampler="uniform",
            explore_sampler="uniform",
            top_frac=0.25,
        )
    )
    tuner.tune(hyper_rose, num_runs=100, n_jobs=n_jobs)
    assert tuner.get_best_scores() == SEED_0_SCORE


@pytest.mark.parametrize("param_config", PARAMS_CONFIG)
@pytest.mark.parametrize("objective_config", OBJECTIVE_CONFIG)
def test_hola(param_config: ParamsSpec, objective_config: ObjectivesSpec) -> None:
    np.random.seed(0)
    exp = HOLA(
        param_config, objective_config, min_samples=30, top_frac=0.25, gmm_sampler="uniform", explore_sampler="uniform"
    )
    assert exp.params_config == PARAMS_CONFIG[0]
    assert exp.objectives_config == OBJECTIVE_CONFIG[0]
    for _ in range(200):
        params = exp.sample()
        value = hyper_rose(**params)
        exp.add_run(value, params)
        # print(_, params, value)
    assert exp.get_best_scores() == SEED_0_SCORE


@pytest.mark.parametrize("param_config", PARAMS_CONFIG)
@pytest.mark.parametrize("objective_config", OBJECTIVE_CONFIG)
def test_tune(param_config: ParamsSpec, objective_config: ObjectivesSpec) -> None:
    tune(hyper_rose, param_config, objective_config)
