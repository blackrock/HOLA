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
from benchmarks.hard_to_optimize_functions.bukin_6_2d import bukin_6, bukin_6_np, bukin_wrapper
from hola.objective import ObjectiveConfig
from hola.params import ParamConfig

PARAMS = {
    "x1": ParamConfig(min=-15, max=-5),
    "x2": ParamConfig(min=-3, max=3),
}
OBJ_CONFIG = {"f": ObjectiveConfig(target=-100, limit=300)}


def hola_bukin(n_iterations: int, config: HolaConfig) -> pat.DataFrame[Analysis]:
    experiment = config.get_hola(PARAMS, OBJ_CONFIG)
    for _ in range(n_iterations):
        suggested_params = experiment.sample()
        res = bukin_6(**suggested_params)
        experiment.add_run({"f": res}, suggested_params)
    best = experiment.get_best_scores()["f"]
    best_params = experiment.get_best_params()
    return Analysis.build_row("bukin6", config.optimizer, config.configuration, n_iterations, best, str(best_params))


def run_hola_bukin(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    return hola_bukin(n_iterations, HolaConfig.hola(n_iterations, configuration, n_dim=2))


def run_sobol_bukin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return hola_bukin(n_iterations, HolaConfig.sobol(n_iterations))


def run_tpe_bukin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_hopt(bukin_wrapper, n_iterations, PARAMS, "bukin6", config=0, is_tpe=True)


def run_random_search_bukin(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    return run_hopt(bukin_wrapper, n_iterations, PARAMS, "bukin6", config=configuration, is_tpe=False)


def run_hooke_bukin(n_iterations: int) -> pat.DataFrame[Analysis]:
    algorithm = PatternSearch()
    return run_pymoo(bukin_6_np, n_iterations, PARAMS, "bukin6", algorithm, config=0)


def run_ga_bukin(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    algorithm = GA(pop_size=GAConfig.build(configuration).pop_size)
    return run_pymoo(bukin_6_np, n_iterations, PARAMS, "bukin6", algorithm, configuration)


def run_nelder_bukin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(bukin_6_np, n_iterations, PARAMS, "bukin6", NelderMead(), 0)


def run_pso_bukin(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(bukin_6_np, n_iterations, PARAMS, "bukin6", PSO(), 0)


BUKIN_OPTIMIZERS: Sequence[Callable[[int], pat.DataFrame[Analysis]]] = [
    # partial(run_hola_bukin, configuration=1),
    partial(run_ga_bukin, configuration=1),
    partial(run_ga_bukin, configuration=2),
    run_nelder_bukin,
    run_pso_bukin,
    run_tpe_bukin,
    run_hooke_bukin,
    partial(run_random_search_bukin, configuration=0),
    partial(run_random_search_bukin, configuration=1),
    run_sobol_bukin,
]


if __name__ == "__main__":
    dataset = ResearchDataset()
    compare_optimizers(BUKIN_OPTIMIZERS, dataset, 50)
    print(dataset.stats)
