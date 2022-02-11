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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.mixture import GaussianMixture


def uniform_to_category(u_sample: float, categories: int) -> int:
    return int(u_sample * categories)


@dataclass  # type: ignore[misc]
class Sampler(ABC):
    dimension: int

    @abstractmethod
    def sample(self) -> npt.NDArray[np.floating]:
        pass

    def fit(self, samples) -> None:
        pass


class UniformSampler(Sampler):
    def sample(self) -> npt.NDArray[np.floating]:
        return np.random.rand(self.dimension)


@dataclass
class SobolSampler(Sampler):
    def __post_init__(self) -> None:
        self.sampler = Sobol(self.dimension)

    def sample(self) -> npt.NDArray[np.floating]:
        return self.sampler.random(1)[0]  # type: ignore[no-any-return]


@dataclass
class GaussianMixtureSampler(Sampler):
    n_components: int
    reg_covar: float
    sample_type: Literal["uniform", "sobol"] = "uniform"

    def __post_init__(self) -> None:
        self.gmm = GaussianMixture(n_components=self.n_components, reg_covar=self.reg_covar)
        self.sobol = Sobol(self.dimension + 1)

    def fit(self, samples) -> None:
        self.gmm.fit(samples)

    def sample(self) -> npt.NDArray[np.floating]:
        return self.uniform_sample() if self.sample_type == "uniform" else self.sobol_sample()

    def uniform_sample(self) -> npt.NDArray[np.floating]:
        sample = self.gmm.sample()[0][0]
        sample[sample > 1] = 1.0
        sample[sample < 0] = 0.0
        return sample  # type: ignore[no-any-return]

    def sobol_sample(self) -> npt.NDArray[np.floating]:
        u_sample = self.sobol.random(1)[0]
        u_component = u_sample[0]
        u_gmm = u_sample[1:]

        component = uniform_to_category(u_component, self.n_components)

        gmm_mean = self.gmm.means_[component]
        gmm_cov = self.gmm.covariances_[component]
        gmm_chol = np.linalg.cholesky(gmm_cov)

        z_gmm = norm.ppf(u_gmm)
        z_gmm = gmm_mean + gmm_chol @ z_gmm
        z_gmm[z_gmm > 1] = 1.0
        z_gmm[z_gmm < 0] = 0.0
        return z_gmm  # type: ignore[no-any-return]


@dataclass
class MixtureSampler(Sampler):
    sampler_explore: Sampler
    sampler_exploit: Sampler | None
    min_explore_samples: int
    min_fit_samples: int
    sample_count: int = field(init=False, default=-1)

    def sample(self) -> npt.NDArray[np.floating]:
        self.sample_count += 1
        explore = self.sample_count <= self.min_explore_samples
        if explore or self.sampler_exploit is None:
            return self.sampler_explore.sample()
        return self.sampler_exploit.sample()

    def fit(self, samples) -> None:
        if samples.ndim == 1:
            return
        if len(samples) < self.min_fit_samples:
            return
        if self.sampler_exploit is None:
            raise TypeError("self.sampler_exploit is None")  # TODO(cmatache): this is a bit strange
        self.sampler_exploit.fit(samples)
