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
from benchmarks.hard_to_optimize_functions.egg_holder_2d import egg_holder, egg_holder_np, egg_holder_wrapper
from hola.objective import ObjectiveConfig
from hola.params import ParamConfig

PARAMS = {"x1": ParamConfig(min=-512, max=512), "x2": ParamConfig(min=-512, max=512)}
OBJ_CONFIG = {"f": ObjectiveConfig(target=-2000, limit=2000)}


def hola_egg_holder(n_iterations: int, config: HolaConfig) -> pat.DataFrame[Analysis]:
    experiment = config.get_hola(PARAMS, OBJ_CONFIG)
    for _ in range(n_iterations):
        suggested_params = experiment.sample()
        res = egg_holder(**suggested_params)
        experiment.add_run({"f": res}, suggested_params)
    best = experiment.get_best_scores()["f"]
    best_params = experiment.get_best_params()
    return Analysis.build_row(
        "egg_holder", config.optimizer, config.configuration, n_iterations, best, str(best_params)
    )


def run_hola_egg_holder(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    return hola_egg_holder(n_iterations, HolaConfig.hola(n_iterations, configuration, n_dim=2))


def run_sobol_egg_holder(n_iterations: int) -> pat.DataFrame[Analysis]:
    return hola_egg_holder(n_iterations, HolaConfig.sobol(n_iterations))


def run_tpe_egg_holder(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_hopt(egg_holder_wrapper, n_iterations, PARAMS, "egg_holder", config=0, is_tpe=True)


def run_random_search_egg_holder(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    return run_hopt(egg_holder_wrapper, n_iterations, PARAMS, "egg_holder", config=configuration, is_tpe=False)


def run_hooke_egg_holder(n_iterations: int) -> pat.DataFrame[Analysis]:
    algorithm = PatternSearch()
    return run_pymoo(egg_holder_np, n_iterations, PARAMS, "egg_holder", algorithm, config=0)


def run_ga_egg_holder(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    algorithm = GA(pop_size=GAConfig.build(configuration).pop_size)
    return run_pymoo(egg_holder_np, n_iterations, PARAMS, "egg_holder", algorithm, configuration)


def run_nelder_egg_holder(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(egg_holder_np, n_iterations, PARAMS, "egg_holder", NelderMead(), 0)


def run_pso_egg_holder(n_iterations: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(egg_holder_np, n_iterations, PARAMS, "egg_holder", PSO(), 0)


EGG_HOLDER_OPTIMIZERS: Sequence[Callable[[int], pat.DataFrame[Analysis]]] = [
    partial(run_ga_egg_holder, configuration=1),
    partial(run_ga_egg_holder, configuration=2),
    # partial(run_hola_egg_holder, configuration=1),
    run_nelder_egg_holder,
    run_pso_egg_holder,
    run_tpe_egg_holder,
    run_hooke_egg_holder,
    partial(run_random_search_egg_holder, configuration=0),
    partial(run_random_search_egg_holder, configuration=1),
    run_sobol_egg_holder,
]

if __name__ == "__main__":
    dataset = ResearchDataset()
    compare_optimizers(
        EGG_HOLDER_OPTIMIZERS,
        dataset,
        2,
    )
    print(dataset.stats)
