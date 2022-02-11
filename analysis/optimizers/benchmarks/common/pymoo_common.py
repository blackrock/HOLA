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

import numpy as np
import numpy.typing as npt
import pandera.typing as pat
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination

from analysis.infra.data_persister import Analysis
from hola.params import ParamConfig


def run_pymoo(
    func: Callable[[Iterable[float]], float] | Callable[[npt.NDArray[np.floating]], float],
    n_iterations: int,
    params: dict[str, ParamConfig],
    benchmark: str,
    algorithm: Algorithm,
    config: int,
) -> pat.DataFrame[Analysis]:
    termination = MaximumFunctionCallTermination(n_iterations)
    problem = FunctionalProblem(
        n_var=len(params),
        objs=[func],
        xl=[param_cfg.min for param_cfg in params.values()],
        xu=[param_cfg.max for param_cfg in params.values()],
    )
    res = minimize(problem, algorithm, termination, verbose=False)
    optimizer = "hooke" if type(algorithm).__name__ == PatternSearch.__name__ else type(algorithm).__name__
    return Analysis.build_row(benchmark, optimizer, config, n_iterations, res.F, str(res.X))
