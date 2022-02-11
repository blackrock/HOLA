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

from typing import Callable, Iterable

import pandera.typing as pat
from hyperopt import fmin, hp, tpe

from analysis.infra.data_persister import Analysis
from hola.params import ParamConfig


def run_hopt(
    func: Callable[[float | Iterable[float]], float] | Callable[[tuple[float, ...]], float],
    n_iterations: int,
    params: dict[str, ParamConfig],
    benchmark: str,
    config: int,
    *,
    is_tpe: bool,
) -> pat.DataFrame[Analysis]:
    space = [hp.uniform(param_name, param_cfg.min, param_cfg.max) for param_name, param_cfg in params.items()]
    if not is_tpe and config == 1:
        n_iterations *= 2
    best_params = fmin(func, space, algo=tpe.suggest if is_tpe else tpe.rand.suggest, max_evals=n_iterations)
    best = func(tuple(best_params.values()))
    optimizer = "tpe" if is_tpe else "random_search"
    return Analysis.build_row(benchmark, optimizer, config, n_iterations, best, str(best_params))
