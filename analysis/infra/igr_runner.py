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

from dataclasses import dataclass
from typing import Callable, cast

import pandera.typing as pat
from pandas import concat

from analysis.infra.data_persister import Analysis
from analysis.optimizers.benchmarks.ackley import ackley
from analysis.optimizers.benchmarks.ackley import get_params as get_ackley_params
from analysis.optimizers.benchmarks.branin import PARAMS as BRANIN_PARAMS
from analysis.optimizers.benchmarks.branin import branin_np
from analysis.optimizers.benchmarks.bukin_6 import PARAMS as BUKIN_PARAMS
from analysis.optimizers.benchmarks.bukin_6 import bukin_6_np
from analysis.optimizers.benchmarks.drop_wave import PARAMS as DROP_WAVE_PARAMS
from analysis.optimizers.benchmarks.drop_wave import drop_wave_np
from analysis.optimizers.benchmarks.egg_holder import PARAMS as EGG_HOLDER_PARAMS
from analysis.optimizers.benchmarks.egg_holder import egg_holder_np
from analysis.optimizers.benchmarks.forrester import PARAMS as FORRESTER_PARAMS
from analysis.optimizers.benchmarks.forrester import forrester_np
from analysis.optimizers.benchmarks.holder_table import PARAMS as HOLDER_TABLE_PARAMS
from analysis.optimizers.benchmarks.holder_table import holder_table_np
from analysis.optimizers.benchmarks.levy_13 import PARAMS as LEVY_13_PARAMS
from analysis.optimizers.benchmarks.levy_13 import levy_13_np
from analysis.optimizers.benchmarks.rastrigin import get_params as get_rastrigin_params
from analysis.optimizers.benchmarks.rastrigin import rastrigin
from analysis.optimizers.benchmarks.schwefel import get_params as get_schwefel_params
from analysis.optimizers.benchmarks.schwefel import schwefel
from analysis.optimizers.benchmarks.six_hump_camel import PARAMS as SIX_HUMP_PARAMS
from analysis.optimizers.benchmarks.six_hump_camel import six_hump_camel_np
from hola.igr_algorithm import IterativeGridRefinement
from hola.params import ParamConfig


@dataclass
class Bench:
    func: Callable
    params: dict[str, ParamConfig]
    name: str


BENCHMARKS = [
    Bench(ackley, get_ackley_params(2), "ackley_2d"),
    Bench(rastrigin, get_rastrigin_params(2), "rastrigin_2d"),
    Bench(schwefel, get_schwefel_params(2), "schwefel_2d"),
    Bench(branin_np, BRANIN_PARAMS, "branin"),
    Bench(bukin_6_np, BUKIN_PARAMS, "bukin6"),
    Bench(drop_wave_np, DROP_WAVE_PARAMS, "drop_wave"),
    Bench(egg_holder_np, EGG_HOLDER_PARAMS, "egg_holder"),
    Bench(six_hump_camel_np, SIX_HUMP_PARAMS, "six_hump_camel"),
    Bench(forrester_np, FORRESTER_PARAMS, "forrester"),
    Bench(holder_table_np, HOLDER_TABLE_PARAMS, "holder_table"),
    Bench(levy_13_np, LEVY_13_PARAMS, "holder_table"),
]


def get_igr_values() -> pat.DataFrame[Analysis]:
    dfs: list[pat.DataFrame[Analysis]] = []
    for bench in BENCHMARKS:
        print(f"Function: {bench.func}")
        tuner = IterativeGridRefinement(bench.params, spacing=4)
        for generations in [1, 2, 3, 4, 8, 12, 20]:
            best = tuner.tune(bench.func, max_generations=generations)
            dfs.append(
                Analysis.build_row(
                    bench.name, "IGR", 0, tuner.get_number_of_iterations(generations), best.val, str(best.params)
                )
            )
        tuner = IterativeGridRefinement(bench.params, spacing=9)
        for generations in [1, 2, 3, 5]:
            best = tuner.tune(bench.func, max_generations=generations)
            dfs.append(
                Analysis.build_row(
                    bench.name, "IGR", 1, tuner.get_number_of_iterations(generations), best.val, str(best.params)
                )
            )
    df = concat(dfs, ignore_index=True)
    return cast(pat.DataFrame[Analysis], df)


if __name__ == "__main__":
    print(get_igr_values().drop(columns=Analysis.best_params))
