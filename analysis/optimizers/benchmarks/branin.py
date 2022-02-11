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

from typing import Callable, Sequence

import pandera.typing as pat
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from returns.curry import partial

from analysis.infra.data_persister import Analysis, ResearchDataset
from analysis.infra.run_helpers import compare_optimizers
from analysis.optimizers.benchmarks.common.hyperopt_common import run_hopt
from analysis.optimizers.benchmarks.common.pymoo_common import run_pymoo
from analysis.optimizers.config.ga_config import GAConfig
from analysis.optimizers.config.hola_config import HolaConfig
from benchmarks.hard_to_optimize_functions.branin_2d import branin, branin_np
from hola.objective import ObjectiveConfig
from hola.params import ParamConfig

PARAMS = {"x1": ParamConfig(min=-5, max=10), "x2": ParamConfig(min=0, max=15)}
OBJ_CONFIG = {"f": ObjectiveConfig(target=-100, limit=400)}


def branin_wrapper(x: tuple[float, ...]) -> float:
    return branin(*x)


def hola_branin(n_iterations: int, config: HolaConfig) -> pat.DataFrame[Analysis]:
    experiment = config.get_hola(PARAMS, OBJ_CONFIG)
    for _ in range(n_iterations):
        suggested_params = experiment.sample()
        res = branin(**suggested_params)
        experiment.add_run({"f": res}, suggested_params)
    best = experiment.get_best_scores()["f"]
    best_params = experiment.get_best_params()
    return Analysis.build_row("branin", config.optimizer, config.configuration, n_iterations, best, str(best_params))


def run_hola_branin(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    return hola_branin(n_iterations, HolaConfig.hola(n_iterations, configuration, n_dim=2))


def run_sobol_branin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return hola_branin(n_iterations, HolaConfig.sobol(n_iterations))


def run_tpe_branin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_hopt(branin_wrapper, n_iterations, PARAMS, "branin", config=0, is_tpe=True)


def run_random_search_branin(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    return run_hopt(branin_wrapper, n_iterations, PARAMS, "branin", config=configuration, is_tpe=False)


def run_hooke_branin(n_iterations: int) -> pat.DataFrame[Analysis]:
    algorithm = PatternSearch()
    return run_pymoo(branin_np, n_iterations, PARAMS, "branin", algorithm, config=0)


def run_ga_branin(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    algorithm = GA(pop_size=GAConfig.build(configuration).pop_size)
    return run_pymoo(branin_np, n_iterations, PARAMS, "branin", algorithm, configuration)


def run_nelder_branin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(branin_np, n_iterations, PARAMS, "branin", NelderMead(), 0)


def run_pso_branin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(branin_np, n_iterations, PARAMS, "branin", PSO(), 0)


BRANIN_OPTIMIZERS: Sequence[Callable[[int], pat.DataFrame[Analysis]]] = [
    partial(run_hola_branin, configuration=49),
    run_sobol_branin,
    run_tpe_branin,
    partial(run_random_search_branin, configuration=0),
    partial(run_random_search_branin, configuration=1),
    run_hooke_branin,
    partial(run_ga_branin, configuration=1),
    run_nelder_branin,
    run_pso_branin,
]


if __name__ == "__main__":
    dataset = ResearchDataset()
    compare_optimizers(BRANIN_OPTIMIZERS, dataset, 50, iterations=(15, 25, 50))
    print(dataset.stats)
