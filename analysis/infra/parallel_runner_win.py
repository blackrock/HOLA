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
from analysis.infra.run_helpers import run_sequentially
from analysis.optimizers.benchmarks.boosted_regressor import (
    run_hola_gboosted_regress,
    run_hooke_gboosted_regress,
    run_nelder_gboosted_regress,
    run_pso_gboosted_regress,
    run_random_search_gboosted_regress,
    run_sobol_gboosted_regress,
    run_tpe_gboosted_regress,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    configs = [49]
    run_sequentially(
        # Validation
        benchmark_optimizers=[
            *[partial(run_hola_gboosted_regress, configuration=cfg) for cfg in configs],
            run_nelder_gboosted_regress,
            run_pso_gboosted_regress,
            run_tpe_gboosted_regress,
            run_hooke_gboosted_regress,
            partial(run_random_search_gboosted_regress, configuration=0),
            partial(run_random_search_gboosted_regress, configuration=1),
            run_sobol_gboosted_regress,
            # # *[partial(run_hola_rosenbrock, configuration=cfg, n_dim=2) for cfg in configs],
            # # *[partial(run_hola_rosenbrock, configuration=cfg, n_dim=5) for cfg in configs],
            # # *[partial(run_hola_rosenbrock, configuration=cfg, n_dim=7) for cfg in configs],
            # *[partial(run_hola_branin, configuration=cfg) for cfg in configs],
            # # *[partial(run_hola_forrester, configuration=cfg) for cfg in configs],
            # # *[partial(run_hola_six_hump_camel, configuration=cfg) for cfg in configs],
            # # *[partial(run_hola_cross_in_tray, configuration=cfg) for cfg in configs],
            # # Other optimizers
            # partial(run_ga_branin, configuration=1),
            # run_nelder_branin,
            # run_pso_branin,
            # run_tpe_branin,
            # run_hooke_branin,
            # partial(run_random_search_branin, configuration=0),
            # partial(run_random_search_branin, configuration=1),
            # run_sobol_branin,
            #
            # partial(run_ga_cross_in_tray, configuration=1),
            # partial(run_ga_cross_in_tray, configuration=2),
            # run_nelder_cross_in_tray,
            # run_pso_cross_in_tray,
            # run_tpe_cross_in_tray,
            # run_hooke_cross_in_tray,
            # partial(run_random_search_cross_in_tray, configuration=0),
            # partial(run_random_search_cross_in_tray, configuration=1),
            # run_sobol_cross_in_tray,
            #
            # partial(run_ga_forrester, configuration=1),
            # run_nelder_forrester,
            # run_pso_forrester,
            # run_tpe_forrester,
            # run_hooke_forrester,
            # partial(run_random_search_forrester, configuration=0),
            # partial(run_random_search_forrester, configuration=1),
            # run_sobol_forrester,
            # *[partial(run_hola_forrester, configuration=cfg) for cfg in configs],
            # #
            # partial(run_ga_six_hump_camel, configuration=1),
            # run_nelder_six_hump_camel,
            # run_pso_six_hump_camel,
            # run_tpe_six_hump_camel,
            # run_hooke_six_hump_camel,
            # partial(run_random_search_six_hump_camel, configuration=0),
            # partial(run_random_search_six_hump_camel, configuration=1),
            # run_sobol_six_hump_camel,
            # *[partial(run_hola_six_hump_camel, configuration=cfg) for cfg in configs],
        ],
        research_dataset=ResearchDataset(),
        n_repetitions=30,
        max_evaluations=(15, 25, 50, 75, 100),  # (15, 25, 50, 75, 100, 200),
    )

# hooke:
#   num iterations 15 requested = 20 in practice
#   num iterations 25 requested = 27 in practice
#   num iterations 50 requested = 52 in practice
#   num iterations 75 requested = 82 in practice
#   num iterations 75 requested = 104 in practice
# pso:
#   num iterations 15 requested = 25 in practice
