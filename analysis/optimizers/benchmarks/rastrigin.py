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
from benchmarks.hard_to_optimize_functions.rastrigin_nd import rastrigin
from hola.objective import ObjectiveConfig
from hola.params import ParamConfig

BOUNDS = ParamConfig(min=-5.12, max=5.12)
OBJ_CONFIG = {"f": ObjectiveConfig(target=-100, limit=100)}


def get_params(n_dim: int = 5) -> dict[str, ParamConfig]:
    return {f"x{i}": BOUNDS for i in range(1, n_dim + 1)}


def hola_rastrigin(n_iterations: int, config: HolaConfig, n_dim: int) -> pat.DataFrame[Analysis]:
    experiment = config.get_hola(get_params(n_dim), OBJ_CONFIG)
    for _ in range(n_iterations):
        suggested_params = experiment.sample()
        res = rastrigin(tuple(suggested_params.values()))
        experiment.add_run({"f": res}, suggested_params)
    best = experiment.get_best_scores()["f"]
    best_params = experiment.get_best_params()
    return Analysis.build_row(
        f"rastrigin_{n_dim}d", config.optimizer, config.configuration, n_iterations, best, str(best_params)
    )


def run_hola_rastrigin(n_iterations: int, configuration: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return hola_rastrigin(n_iterations, HolaConfig.hola(n_iterations, configuration, n_dim), n_dim)


def run_sobol_rastrigin(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return hola_rastrigin(n_iterations, HolaConfig.sobol(n_iterations), n_dim)


def run_tpe_rastrigin(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return run_hopt(rastrigin, n_iterations, get_params(n_dim), f"rastrigin_{n_dim}d", config=0, is_tpe=True)


def run_random_search_rastrigin(n_iterations: int, n_dim: int, configuration: int) -> pat.DataFrame[Analysis]:
    return run_hopt(
        rastrigin, n_iterations, get_params(n_dim), f"rastrigin_{n_dim}d", config=configuration, is_tpe=False
    )


def run_hooke_rastrigin(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    algorithm = PatternSearch()
    return run_pymoo(rastrigin, n_iterations, get_params(n_dim), f"rastrigin_{n_dim}d", algorithm, config=0)


def run_ga_rastrigin(n_iterations: int, n_dim: int, configuration: int) -> pat.DataFrame[Analysis]:
    algorithm = GA(pop_size=GAConfig.build(configuration).pop_size)
    return run_pymoo(rastrigin, n_iterations, get_params(n_dim), f"rastrigin_{n_dim}d", algorithm, configuration)


def run_nelder_rastrigin(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(rastrigin, n_iterations, get_params(n_dim), f"rastrigin_{n_dim}d", NelderMead(), 0)


def run_pso_rastrigin(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(rastrigin, n_iterations, get_params(n_dim), f"rastrigin_{n_dim}d", PSO(), 0)


RASTRIGIN_OPTIMIZERS: Sequence[Callable[[int], pat.DataFrame[Analysis]]] = [
    partial(run_ga_rastrigin, configuration=1, n_dim=5),
    partial(run_ga_rastrigin, configuration=2, n_dim=5),
    partial(run_nelder_rastrigin, n_dim=5),
    partial(run_pso_rastrigin, n_dim=5),
    # partial(run_hola_rastrigin, configuration=1, n_dim=5),
    partial(run_tpe_rastrigin, n_dim=5),
    partial(run_hooke_rastrigin, n_dim=5),
    partial(run_random_search_rastrigin, n_dim=5, configuration=0),
    partial(run_random_search_rastrigin, n_dim=5, configuration=1),
    partial(run_sobol_rastrigin, n_dim=5),
]


if __name__ == "__main__":
    dataset = ResearchDataset()
    compare_optimizers(
        RASTRIGIN_OPTIMIZERS,
        dataset,
        50,
    )
    print(dataset.stats)
