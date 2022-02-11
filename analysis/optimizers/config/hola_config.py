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
from typing import Type, TypeVar

from typing_extensions import Literal

from hola.algorithm import HOLA
from hola.objective import ObjectivesSpec
from hola.params import ParamsSpec
from hola.tune import safe_n_components

THolaConfig = TypeVar("THolaConfig", bound="HolaConfig")


@dataclass
class HolaConfig:  # pylint: disable=too-many-instance-attributes
    min_samples: int | None
    configuration: int
    optimizer: Literal["hola", "sobol"]
    gmm_sampler: Literal["uniform", "sobol"]
    explore_sampler: Literal["uniform", "sobol"]
    n_components: int
    min_fit_samples: int
    top_frac: float
    gmm_reg: float

    def get_hola(self, params: ParamsSpec, objs: ObjectivesSpec) -> HOLA:
        return HOLA(
            params,
            objs,
            min_samples=self.min_samples,
            gmm_sampler=self.gmm_sampler,
            explore_sampler=self.explore_sampler,
            n_components=self.n_components,
            min_fit_samples=self.min_fit_samples,
            gmm_reg=self.gmm_reg,
            top_frac=self.top_frac,
        )

    @classmethod
    def sobol(cls: Type[THolaConfig], n_iterations: int) -> THolaConfig:
        return cls(
            min_samples=n_iterations,
            configuration=0,
            optimizer="sobol",
            gmm_sampler="sobol",
            explore_sampler="sobol",
            n_components=3,
            min_fit_samples=5,
            top_frac=0.25,
            gmm_reg=0.0005,
        )

    @classmethod
    def hola(cls: Type[THolaConfig], n_iterations: int, configuration: int, n_dim: int) -> THolaConfig:
        # pylint: disable=too-many-branches,too-many-return-statements,too-many-statements
        if configuration == 1:
            return cls(
                min_samples=max(n_dim**2, 20),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="sobol",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 2:
            return cls(
                min_samples=n_iterations // 3,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="sobol",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 3:
            return cls(
                min_samples=n_iterations // 2,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="sobol",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 4:
            return cls(
                min_samples=max(n_dim**2, 20),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 5:
            return cls(
                min_samples=n_iterations // 3,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 6:
            return cls(
                min_samples=max(n_dim**2, 20),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="uniform",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 7:
            return cls(
                min_samples=n_iterations // 3,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="uniform",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 8:
            return cls(
                min_samples=50,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 9:  # bad
            return cls(
                min_samples=max(n_dim**2, 20),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=4,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 10:  # bad
            return cls(
                min_samples=max(n_dim**2, 20),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=5,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 11:
            return cls(
                min_samples=60,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 12:
            return cls(
                min_samples=60,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="sobol",
                explore_sampler="sobol",
                n_components=3,
                min_fit_samples=5,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 13:
            return cls(
                min_samples=60,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 14:
            return cls(
                min_samples=100,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 15:
            return cls(
                min_samples=n_iterations // 3,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 16:
            return cls(
                min_samples=60,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=10,
                min_fit_samples=10,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 17:
            return cls(
                min_samples=60,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=15,
                min_fit_samples=15,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 18:
            return cls(
                min_samples=25,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 19:
            return cls(
                min_samples=60,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="sobol",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 20:
            return cls(
                min_samples=n_dim * 30,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 21:
            return cls(
                min_samples=n_dim * 20,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 22:
            return cls(
                min_samples=n_dim * 20,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.5,
                gmm_reg=0.0005,
            )
        if configuration == 23:
            return cls(
                min_samples=10 * n_dim + 20,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 24:
            return cls(
                min_samples=10 * n_dim + 20,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.5,
                gmm_reg=0.0005,
            )
        if configuration == 25:
            return cls(
                min_samples=min(n_iterations // 4, 10 * n_dim + 30),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 26:
            return cls(
                min_samples=min(n_iterations // 4, 10 * n_dim + 30),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.33,
                gmm_reg=0.0005,
            )
        if configuration == 27:
            return cls(
                min_samples=min(n_iterations // 4, 10 * n_dim + 30),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.5,
                gmm_reg=0.0005,
            )
        if configuration == 28:
            return cls(
                min_samples=60,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.33,
                gmm_reg=0.0005,
            )
        if configuration == 29:
            return cls(
                min_samples=min(n_iterations // 4, 5 * n_dim + 35),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 30:
            return cls(
                min_samples=min(n_iterations // 4, 5 * n_dim + 35),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=7,
                min_fit_samples=7,
                top_frac=0.33,
                gmm_reg=0.0005,
            )
        if configuration == 31:
            return cls(
                min_samples=max(min(n_iterations // 4, 10 * n_dim + 30), 60),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=15,
                min_fit_samples=15,
                top_frac=0.33,
                gmm_reg=0.0005,
            )
        if configuration == 32:
            return cls(
                min_samples=max(min(n_iterations // 4, 10 * n_dim + 30), 60),
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=15,
                min_fit_samples=15,
                top_frac=0.25,
                gmm_reg=0.0005,
            )
        if configuration == 33:
            min_samples = get_min_samples_33(n_iterations, n_dim)
            top_frac = 0.25
            n_components = get_n_components(n_dim, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 34:
            top_frac = 0.25
            min_samples = get_min_samples_33(n_iterations, n_dim)
            n_components = safe_n_components(15, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 35:
            top_frac = 0.25
            min_samples = get_min_samples_33(n_iterations, n_dim)
            n_components = safe_n_components(15, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.001,
            )
        if configuration == 36:
            top_frac = 0.25
            min_samples = get_min_samples_33(n_iterations, n_dim)
            n_components = safe_n_components(20, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.001,
            )
        # pq_data_3
        if configuration == 37:
            top_frac = 0.25
            min_samples = 40
            n_components = safe_n_components(7, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 38:
            top_frac = 0.25
            min_samples = 60
            n_components = safe_n_components(7, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 39:
            top_frac = 0.25
            min_samples = 40
            n_components = safe_n_components(15, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 40:
            top_frac = 0.25
            min_samples = 60
            n_components = safe_n_components(15, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 41:
            top_frac = 0.25
            min_samples = 60
            n_components = safe_n_components(3, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 42:
            top_frac = 0.25
            min_samples = 40
            n_components = safe_n_components(3, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 43:
            top_frac = 0.2
            min_samples = 60
            n_components = safe_n_components(7, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 44:
            top_frac = 0.2
            min_samples = 100
            n_components = safe_n_components(7, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 45:
            top_frac = 0.25
            min_samples = 20
            n_components = safe_n_components(7, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 46:
            top_frac = 0.25
            min_samples = 20
            n_components = safe_n_components(15, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 47:
            top_frac = 0.25
            min_samples = 20
            n_components = safe_n_components(3, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 48:
            top_frac = 0.25
            min_samples = min(n_iterations // 5, 2 * n_dim + 50)
            n_components = safe_n_components(n_dim + 1, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 49:
            top_frac = 0.2
            min_samples = min(n_iterations // 5, 2 * n_dim + 50)
            n_components = safe_n_components(n_dim + 1, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 50:
            top_frac = 0.25
            min_samples = min(n_iterations // 5, 2 * n_dim + 50)
            n_components = safe_n_components(2 * n_dim, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        if configuration == 51:
            top_frac = 0.2
            min_samples = min(n_iterations // 5, 2 * n_dim + 50)
            n_components = safe_n_components(2 * n_dim, min_samples, top_frac)
            return cls(
                min_samples=min_samples,
                configuration=configuration,
                optimizer="hola",
                gmm_sampler="uniform",
                explore_sampler="sobol",
                n_components=n_components,
                min_fit_samples=n_components,
                top_frac=top_frac,
                gmm_reg=0.0005,
            )
        raise ValueError("Unknown HOLA configuration")


def get_min_samples_33(n_iterations: int, n_dim: int) -> int:
    return min(
        n_iterations // 4,  # If n_iterations is small -> only explore for a quarter, if it's too large use n_dim
        2 * n_dim + 50,  # The more dimensions the more initial exploration
    )


def get_n_components(n_dim: int, min_samples: int, top_frac: float) -> int:
    return safe_n_components(n_dim + 10, min_samples, top_frac)
