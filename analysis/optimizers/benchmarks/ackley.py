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
from skopt import forest_minimize

from analysis.infra.data_persister import Analysis, ResearchDataset
from analysis.infra.run_helpers import compare_optimizers
from analysis.optimizers.benchmarks.common.hyperopt_common import run_hopt
from analysis.optimizers.benchmarks.common.pymoo_common import run_pymoo
from analysis.optimizers.config.ga_config import GAConfig
from analysis.optimizers.config.hola_config import HolaConfig
from benchmarks.hard_to_optimize_functions.ackley_nd import ackley
from hola.objective import ObjectiveConfig
from hola.params import ParamConfig

BOUNDS = ParamConfig(min=-32.768, max=32.768)


def get_params(n_dim: int = 5) -> dict[str, ParamConfig]:
    return {f"x{i}": BOUNDS for i in range(1, n_dim + 1)}


OBJ_CONFIG = {"f": ObjectiveConfig(target=-100, limit=2000)}


def hola_ackley(n_iterations: int, config: HolaConfig, n_dim: int) -> pat.DataFrame[Analysis]:
    experiment = config.get_hola(get_params(n_dim), OBJ_CONFIG)
    for _ in range(n_iterations):
        suggested_params = experiment.sample()
        res = ackley(tuple(suggested_params.values()))
        experiment.add_run({"f": res}, suggested_params)
    best = experiment.get_best_scores()["f"]
    best_params = experiment.get_best_params()
    return Analysis.build_row(
        f"ackley_{n_dim}d", config.optimizer, config.configuration, n_iterations, best, str(best_params)
    )


def run_hola_ackley(n_iterations: int, configuration: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return hola_ackley(n_iterations, HolaConfig.hola(n_iterations, configuration, n_dim), n_dim)


def run_sobol_ackley(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return hola_ackley(n_iterations, HolaConfig.sobol(n_iterations), n_dim)


def run_tpe_ackley(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return run_hopt(ackley, n_iterations, get_params(n_dim), f"ackley_{n_dim}d", config=0, is_tpe=True)


def run_random_search_ackley(n_iterations: int, n_dim: int, configuration: int) -> pat.DataFrame[Analysis]:
    return run_hopt(ackley, n_iterations, get_params(n_dim), f"ackley_{n_dim}d", config=configuration, is_tpe=False)


def run_hooke_ackley(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    algorithm = PatternSearch()
    return run_pymoo(ackley, n_iterations, get_params(n_dim), f"ackley_{n_dim}d", algorithm, config=0)


def run_ga_ackley(n_iterations: int, n_dim: int, configuration: int) -> pat.DataFrame[Analysis]:
    algorithm = GA(pop_size=GAConfig.build(configuration).pop_size)
    return run_pymoo(ackley, n_iterations, get_params(n_dim), f"ackley_{n_dim}d", algorithm, configuration)


def run_nelder_ackley(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(ackley, n_iterations, get_params(n_dim), f"ackley_{n_dim}d", NelderMead(), 0)


def run_pso_ackley(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    return run_pymoo(ackley, n_iterations, get_params(n_dim), f"ackley_{n_dim}d", PSO(), 0)


def run_forest_minimize_ackley(n_iterations: int, n_dim: int) -> pat.DataFrame[Analysis]:
    params = get_params(n_dim)
    bests = forest_minimize(
        ackley,
        [(param_cfg.min, param_cfg.max) for _, param_cfg in params.items()],
        n_calls=n_iterations,
    )
    return Analysis.build_row(f"ackley_{n_dim}d", "skopt_forest", 0, n_iterations, ackley(bests.x), str(bests.x))


ACKLEY_OPTIMIZERS: Sequence[Callable[[int], pat.DataFrame[Analysis]]] = [
    partial(run_ga_ackley, configuration=1, n_dim=5),
    partial(run_ga_ackley, configuration=2, n_dim=5),
    # partial(run_forest_minimize_ackley, n_dim=5),
    # partial(run_hola_ackley, configuration=13, n_dim=5),
    partial(run_nelder_ackley, n_dim=5),
    partial(run_pso_ackley, n_dim=5),
    partial(run_tpe_ackley, n_dim=5),
    partial(run_hooke_ackley, n_dim=5),
    partial(run_random_search_ackley, n_dim=5, configuration=0),
    partial(run_random_search_ackley, n_dim=5, configuration=1),
    partial(run_sobol_ackley, n_dim=5),
]


if __name__ == "__main__":
    dataset = ResearchDataset()
    compare_optimizers(ACKLEY_OPTIMIZERS, dataset, 50)
    print(dataset.stats)
