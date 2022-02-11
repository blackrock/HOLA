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

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from hola.objective import (
    FuncName,
    ObjectiveConfig,
    ObjectiveConfigSpec,
    ObjectiveScalarizer,
    ObjectivesSpec,
    parse_objectives_config,
)
from hola.params import ParameterTransformer, ParamName, ParamsSpec, parse_params_config
from hola.sample import GaussianMixtureSampler, MixtureSampler, Sampler, SobolSampler, UniformSampler


class Leaderboard:
    """The Leaderboard keeps track of all the parameters tried and their corresponding scores."""

    def __init__(
        self,
        param_transformer: ParameterTransformer,
        objective_scalarizer: ObjectiveScalarizer,
    ):
        self.param_transformer = param_transformer
        self.objective_scalarizer = objective_scalarizer

        # Parameters tried so far (param name -> param value)
        self.params: list[dict[ParamName, float]] = []
        self.params_u: list[npt.NDArray[np.floating]] = []
        # Objectives seen so far (correspond piece-wise with self.params)
        self.objectives: list[dict[FuncName, float]] = []
        self.scores: list[float] = []
        self.sorted_keys: list[int] | npt.NDArray[np.integer] = []

    def add_row(
        self,
        u_params: npt.NDArray[np.floating],
        params: dict[ParamName, float],
        objectives: dict[FuncName, float],
        score: float,
    ) -> None:
        self.objectives.append(objectives)
        self.params.append(params)
        self.params_u.append(u_params)
        self.scores.append(score)
        self.sort_samples()

    def sort_samples(self) -> None:
        score_tups: list[tuple[int, float]] = list(enumerate(self.scores))
        score_tups = sorted(score_tups, key=lambda t: t[1])
        self.sorted_keys = [t[0] for t in score_tups]

    def get_elite_samples(self, elite_fraction: float) -> npt.NDArray[np.floating]:
        n_samples = self.num_samples()
        n_elite = int(elite_fraction * n_samples)
        if n_elite <= 1:
            return np.array(self.params_u)
        u_list = [self.params_u[i] for i in self.sorted_keys[:n_elite]]
        return np.array(u_list)

    def get_u_params(self, index: int) -> npt.NDArray[np.floating]:
        return np.array(self.params_u[index])

    def get_params(self, index: int) -> dict[ParamName, float]:
        return self.params[index]

    def get_best_params(self) -> dict[ParamName, float]:
        return self.params[self.sorted_keys[0]]

    def get_objectives(self, index: int) -> dict[FuncName, float]:
        return self.objectives[index]

    def get_best_objectives(self) -> dict[FuncName, float]:
        return self.objectives[self.sorted_keys[0]]

    def get_score(self, index: int) -> float:
        return self.scores[index]

    def get_best_score(self) -> float:
        return self.scores[self.sorted_keys[0]]

    def set_score(self, index: int, score: float) -> None:
        self.scores[index] = score

    def recompute_scores(self) -> None:
        for i, obj in enumerate(self.objectives):
            score = self.objective_scalarizer.scalarize_objectives(obj)
            self.scores[i] = score

    def num_samples(self) -> int:
        return len(self.scores)

    def get_dataframe(self) -> pd.DataFrame:
        rows = []

        for i in range(self.num_samples()):
            d: list[float] = [i]

            for param_name in self.param_transformer.params_config:
                d.append(self.params[i][param_name])
            for objective_name in self.objective_scalarizer.objectives_config:
                d.append(self.objectives[i][objective_name])
            d.append(self.scores[i])

            rows.append(d)

        cols = ["run"]
        for param_name in self.param_transformer.params_config:
            cols.append(param_name)
        for objective_name in self.objective_scalarizer.objectives_config.keys():
            cols.append(objective_name)
        cols.append("score")

        df = pd.DataFrame(data=rows, columns=cols)
        return df

    def get_leaderboard(self) -> pd.DataFrame:
        df = self.get_dataframe()
        return df.sort_values("score", ascending=True)

    def save(self, filename: Path | str) -> None:
        df = self.get_dataframe()
        df.to_csv(filename)

    def restore(self, df: pd.DataFrame) -> None:
        param_keys = list(self.param_transformer.params_config)
        obj_keys = list(self.objective_scalarizer.objectives_config)

        param_vals = df[param_keys].values
        obj_vals = df[obj_keys].values

        n = param_vals.shape[0]

        self.objectives = [{}] * n  # TODO(cmatache) typing seems to reveal code smell
        self.params = [{}] * n
        self.params_u = [np.array([])] * n
        self.scores = [0] * n

        self.sorted_keys = df["run"].values

        for i, key in enumerate(self.sorted_keys):
            d = {}
            for j, c in enumerate(param_keys):
                d[c] = param_vals[i][j]
            self.params[key] = d
            # TODO types may not be okay on the next line
            self.params_u[key] = self.param_transformer.back_transform_param_dict(d)

        for i, key in enumerate(self.sorted_keys):
            d = {}
            for j, c in enumerate(obj_keys):
                d[c] = obj_vals[i][j]
            self.objectives[key] = d

        self.recompute_scores()
        self.sort_samples()

    def load(self, filename: Path | str) -> None:
        df = pd.read_csv(filename, index_col=0)
        self.restore(df)


class HOLA:  # pylint: disable=too-many-instance-attributes
    """HOLA algorithm, combines the Leaderboard with sampling methods."""

    def __init__(
        self,
        params_config: ParamsSpec,
        objectives_config: ObjectivesSpec,
        top_frac: float = 0.2,
        min_samples: int | None = None,
        min_fit_samples: int | None = None,
        n_components: int = 3,
        gmm_reg: float = 0.0005,
        gmm_sampler: Literal["uniform", "sobol"] = "uniform",
        explore_sampler: Literal["uniform", "sobol"] = "sobol",
    ):
        # Parse parameters
        self.params_config = parse_params_config(params_config)
        num_params = len(self.params_config)
        self.objectives_config = parse_objectives_config(objectives_config)
        # Scaling utils
        self.param_transformer = ParameterTransformer(self.params_config)
        self.objective_scalarizer = ObjectiveScalarizer(self.objectives_config)
        self.leaderboard = Leaderboard(self.param_transformer, self.objective_scalarizer)
        # Sampling params
        self.min_samples = max((num_params ** 2) if min_samples is None else min_samples, 10 * num_params)
        self.top_frac = top_frac
        # Samplers
        self.sampler_explore: Sampler
        if explore_sampler == "uniform":
            self.sampler_explore = UniformSampler(dimension=num_params)
        else:
            self.sampler_explore = SobolSampler(dimension=num_params)
        self.sampler_exploit = GaussianMixtureSampler(
            dimension=num_params, n_components=n_components, reg_covar=gmm_reg, sample_type=gmm_sampler
        )
        self.min_fit_samples = max(min_fit_samples if min_fit_samples is not None else n_components, 2)
        assert int(self.min_samples * self.top_frac) >= self.min_fit_samples  # We need to fit before we can GMM sample
        self.dist = MixtureSampler(
            dimension=num_params,
            sampler_explore=self.sampler_explore,
            sampler_exploit=self.sampler_exploit,
            min_explore_samples=self.min_samples,
            min_fit_samples=self.min_fit_samples,
        )

    def add_run(self, objectives: dict[FuncName, float], params: dict[ParamName, float]) -> None:
        """Let the HOLA algorithm know about the latest parameters tried and the values of the objectives."""
        u_params = self.param_transformer.back_transform_param_dict(params)
        score = self.objective_scalarizer.scalarize_objectives(objectives)
        self.leaderboard.add_row(u_params, params, objectives, score)
        self.fit()

    def set_objectives(self, obj_config: dict[FuncName, ObjectiveConfig] | dict[FuncName, ObjectiveConfigSpec]) -> None:
        """Set Objectives externally (used in the server)."""
        self.objectives_config = parse_objectives_config(obj_config)
        self.objective_scalarizer = ObjectiveScalarizer(self.objectives_config)
        self.leaderboard.objective_scalarizer = self.objective_scalarizer
        self.leaderboard.recompute_scores()
        self.fit()

    def fit(self) -> None:
        """Fir the Gaussian Mixture model over the elite samples."""
        elite_samples = self.leaderboard.get_elite_samples(self.top_frac)
        self.dist.fit(elite_samples)

    def sample(self) -> dict[ParamName, float]:
        """Suggest new hyperparameters."""
        sample: npt.NDArray[np.floating]
        sample = self.dist.sample()
        # Scale params
        params = self.param_transformer.transform_u_params(sample)
        return params

    def get_best_params(self) -> dict[ParamName, float]:
        return self.leaderboard.get_best_params()

    def get_best_scores(self) -> dict[FuncName, float]:
        return self.leaderboard.get_best_objectives()

    def get_leaderboard(self) -> pd.DataFrame:
        return self.leaderboard.get_leaderboard()

    def save(self, file: Path | str) -> None:
        self.leaderboard.save(file)

    def load(self, file: Path | str) -> None:
        self.leaderboard.load(file)
        self.fit()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"top_frac={self.top_frac}, min_samples={self.min_samples}, min_fit_samples={self.min_fit_samples}, "
            f"n_components={self.sampler_exploit.n_components}, gmm_reg={self.sampler_exploit.reg_covar}, "
            f"gmm_sampler='{self.sampler_exploit.sample_type}', explore_sampler={self.sampler_explore},"
            f"params_config={self.params_config}, objectives_config={self.objectives_config})"
        )
