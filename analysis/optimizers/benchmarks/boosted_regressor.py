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

from functools import partial
from typing import Callable, Iterable, Sequence, cast

import pandera.typing as pat
from hyperopt import fmin, hp, tpe
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.factory import get_sampling
from pymoo.operators.mixed_variable_operator import MixedVariableSampling
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from analysis.infra.data_persister import Analysis
from analysis.optimizers.config.hola_config import HolaConfig
from hola.objective import ObjectiveConfig
from hola.params import ParamConfig

HOPT_SPACE = [
    hp.choice("n_estimators", range(10, 1001)),
    hp.choice("max_depth", [1, 2, 3, 4]),
    hp.uniform("learning_rate", 1e-4, 1),
    hp.uniform("subsample", 0.2, 1),
]

PYMOO_SAMPLING = MixedVariableSampling(
    ["n_estimators", "max_depth", "learning_rate", "subsample"],
    {
        "n_estimators": get_sampling("int_random"),
        "max_depth": get_sampling("int_random"),
        "learning_rate": get_sampling("real_random"),
        "subsample": get_sampling("real_random"),
    },
)

PARAMS = {
    "n_estimators": ParamConfig(min=10, max=1000, param_type="int", scale="log", grid=20),
    "max_depth": ParamConfig(values=[1, 2, 3, 4]),
    "learning_rate": ParamConfig(min=1e-4, max=1, scale="log"),
    "subsample": ParamConfig(min=0.2, max=1),
}

# define objectives
OBJ_CONFIG = {
    "r_squared": ObjectiveConfig(target=-1, limit=1000, priority=2),
    # "abs_error": ObjectiveConfig(target=0, limit=1000, priority=0.5),
}

BENCH_NAME = "gboosted_regress"
DATASET = datasets.load_diabetes()
DATA, TARGET = DATASET.data, DATASET.target
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(DATA, TARGET, test_size=0.1, random_state=13)


def gboosted_regress(params: Iterable[float]) -> float:
    n_estimators, max_depth, learning_rate, subsample = params
    model = GradientBoostingRegressor(
        n_estimators=int(n_estimators), max_depth=int(max_depth), learning_rate=learning_rate, subsample=subsample
    )
    model.fit(X_TRAIN, Y_TRAIN)
    r2 = model.score(X_TEST, Y_TEST)
    # abs_err = np.mean(np.abs(model.predict(X_TEST) - Y_TEST))
    return cast(float, -r2)  # pylint: disable=invalid-unary-operand-type


PYMOO_PROBLEM = FunctionalProblem(
    n_var=len(PARAMS),
    objs=[gboosted_regress],
    xl=[10, 1, 1e-4, 0.2],
    xu=[1000, 4, 1, 1],
)


def hola_gboosted_regress(n_iterations: int, config: HolaConfig) -> pat.DataFrame[Analysis]:
    experiment = config.get_hola(PARAMS, OBJ_CONFIG)
    for _ in range(n_iterations):
        suggested_params = experiment.sample()
        res = gboosted_regress(tuple(suggested_params.values()))
        experiment.add_run({"r_squared": res}, suggested_params)
    best_r2 = experiment.get_best_scores()["r_squared"]
    # best_err = experiment.get_best_scores()["abs_error"]
    best_params = experiment.get_best_params()
    return Analysis.build_row(
        BENCH_NAME, config.optimizer, config.configuration, n_iterations, best_r2, str(best_params)
    )


def run_hola_gboosted_regress(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    return hola_gboosted_regress(n_iterations, HolaConfig.hola(n_iterations, configuration, n_dim=4))


def run_sobol_gboosted_regress(n_iterations: int) -> pat.DataFrame[Analysis]:
    return hola_gboosted_regress(n_iterations, HolaConfig.sobol(n_iterations))


def run_tpe_gboosted_regress(n_iterations: int) -> pat.DataFrame[Analysis]:
    best_params = fmin(gboosted_regress, HOPT_SPACE, algo=tpe.suggest, max_evals=n_iterations, return_argmin=False)
    best = gboosted_regress(best_params)
    return Analysis.build_row(BENCH_NAME, "tpe", 0, n_iterations, best, str(best_params))


def run_random_search_gboosted_regress(n_iterations: int, configuration: int) -> pat.DataFrame[Analysis]:
    if configuration == 1:
        n_iterations *= 2
    best_params = fmin(gboosted_regress, HOPT_SPACE, algo=tpe.rand.suggest, max_evals=n_iterations, return_argmin=False)
    best = gboosted_regress(best_params)
    return Analysis.build_row(BENCH_NAME, "random_search", configuration, n_iterations, best, str(best_params))


def run_hooke_gboosted_regress(n_iterations: int) -> pat.DataFrame[Analysis]:
    algorithm = PatternSearch(sampling=PYMOO_SAMPLING)
    termination = MaximumFunctionCallTermination(n_iterations)
    res = minimize(PYMOO_PROBLEM, algorithm, termination, verbose=True)
    return Analysis.build_row(BENCH_NAME, "hooke", 0, n_iterations, res.F, str(res.X))


def run_nelder_gboosted_regress(n_iterations: int) -> pat.DataFrame[Analysis]:
    # concatenate_mixed_variables: concats int and float np.ndarrays into ndarray with dtype object
    # which cannot be checked with .isna() in max_alpha (vectors.py)
    algorithm = NelderMead(sampling=PYMOO_SAMPLING)
    termination = MaximumFunctionCallTermination(n_iterations)
    res = minimize(PYMOO_PROBLEM, algorithm, termination, verbose=True)
    optimizer = type(algorithm).__name__
    return Analysis.build_row(BENCH_NAME, optimizer, 0, n_iterations, res.F, str(res.X))


def run_pso_gboosted_regress(n_iterations: int) -> pat.DataFrame[Analysis]:
    algorithm = PSO(sampling=PYMOO_SAMPLING)
    termination = MaximumFunctionCallTermination(n_iterations)
    res = minimize(PYMOO_PROBLEM, algorithm, termination, verbose=True)
    optimizer = type(algorithm).__name__
    return Analysis.build_row(BENCH_NAME, optimizer, 0, n_iterations, res.F, str(res.X))


GBOOSTED_REGRESS_OPTIMIZERS: Sequence[Callable[[int], pat.DataFrame[Analysis]]] = [
    # partial(run_hola_gboosted_regress, configuration=1),
    partial(run_random_search_gboosted_regress, configuration=0),
    partial(run_random_search_gboosted_regress, configuration=1),
    run_sobol_gboosted_regress,
]

if __name__ == "__main__":
    # run_tpe_gboosted_regress(50)
    run_pso_gboosted_regress(50)
    # run_nelder_gboosted_regress(25)
    # print(run_hooke_gboosted_regress(100))
    # print(run_tpe_gboosted_regress(15))
    # print(run_hola_gboosted_regress(15, 49))
