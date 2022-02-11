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
import logging

from returns.curry import partial

from analysis.infra.data_persister import ResearchDataset
from analysis.infra.run_helpers import run_parallel_by_optimizer
from analysis.optimizers.benchmarks.ackley import run_hola_ackley, run_nelder_ackley, run_pso_ackley, run_sobol_ackley
from analysis.optimizers.benchmarks.bukin_6 import run_hola_bukin, run_nelder_bukin, run_pso_bukin, run_sobol_bukin
from analysis.optimizers.benchmarks.drop_wave import (
    run_hola_drop_wave,
    run_nelder_drop_wave,
    run_pso_drop_wave,
    run_sobol_drop_wave,
)
from analysis.optimizers.benchmarks.egg_holder import (
    run_hola_egg_holder,
    run_nelder_egg_holder,
    run_pso_egg_holder,
    run_sobol_egg_holder,
)
from analysis.optimizers.benchmarks.holder_table import (
    run_hola_holder_table,
    run_nelder_holder_table,
    run_pso_holder_table,
    run_sobol_holder_table,
)
from analysis.optimizers.benchmarks.levy_13 import (
    run_hola_levy_13,
    run_nelder_levy_13,
    run_pso_levy_13,
    run_sobol_levy_13,
)
from analysis.optimizers.benchmarks.rastrigin import (
    run_hola_rastrigin,
    run_nelder_rastrigin,
    run_pso_rastrigin,
    run_sobol_rastrigin,
)
from analysis.optimizers.benchmarks.schwefel import (
    run_hola_schwefel,
    run_nelder_schwefel,
    run_pso_schwefel,
    run_sobol_schwefel,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_parallel_by_optimizer(
        benchmark_optimizers=[
            partial(run_nelder_ackley, n_dim=5),
            partial(run_pso_ackley, n_dim=5),
            run_nelder_bukin,
            run_pso_bukin,
            run_nelder_drop_wave,
            run_pso_drop_wave,
            run_nelder_egg_holder,
            run_pso_egg_holder,
            run_nelder_holder_table,
            run_pso_holder_table,
            run_nelder_levy_13,
            run_pso_levy_13,
            partial(run_nelder_rastrigin, n_dim=5),
            partial(run_pso_rastrigin, n_dim=5),
            partial(run_nelder_schwefel, n_dim=5),
            partial(run_pso_schwefel, n_dim=5),
        ],
        research_dataset=ResearchDataset(),
        n_repetitions=10,
        n_processes=3,
    )
    # run_parallel_by_optimizer(
    #     benchmark_optimizers=[
    #         *ACKLEY_OPTIMIZERS,
    #         *BUKIN_OPTIMIZERS,
    #         *DROP_WAVE_OPTIMIZERS,
    #         *EGG_HOLDER_OPTIMIZERS,
    #         *HOLDER_TABLE_OPTIMIZERS,
    #         *LEVY_13_OPTIMIZERS,
    #         *RASTRIGIN_OPTIMIZERS,
    #         *SCHWEFEL_OPTIMIZERS,
    #     ],
    #     research_dataset=ResearchDataset(),
    #     n_repetitions=30,
    #     n_processes=10,
    # )
    validation_configs = list(range(17, 0, -1))
    run_parallel_by_optimizer(
        # Validation
        benchmark_optimizers=[
            *[partial(run_hola_ackley, configuration=cfg, n_dim=5) for cfg in validation_configs],
            *[partial(run_hola_bukin, configuration=cfg) for cfg in validation_configs],
            *[partial(run_hola_drop_wave, configuration=cfg) for cfg in validation_configs],
            *[partial(run_hola_egg_holder, configuration=cfg) for cfg in validation_configs],
            *[partial(run_hola_holder_table, configuration=cfg) for cfg in validation_configs],
            *[partial(run_hola_levy_13, configuration=cfg) for cfg in validation_configs],
            *[partial(run_hola_rastrigin, configuration=cfg, n_dim=5) for cfg in validation_configs],
            *[partial(run_hola_schwefel, configuration=cfg, n_dim=5) for cfg in validation_configs],
        ],
        research_dataset=ResearchDataset(),
        n_repetitions=15,
        n_processes=2,
    )
    run_parallel_by_optimizer(
        benchmark_optimizers=[
            partial(run_sobol_ackley, n_dim=5),
            run_sobol_bukin,
            run_sobol_drop_wave,
            run_sobol_egg_holder,
            run_sobol_holder_table,
            run_sobol_levy_13,
            partial(run_sobol_rastrigin, n_dim=5),
            partial(run_sobol_schwefel, n_dim=5),
        ],
        research_dataset=ResearchDataset(),
        n_repetitions=10,
        n_processes=2,
    )
