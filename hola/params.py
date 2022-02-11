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
"""params transformer.

internally we sample params as just U(0,1) random variables

These are then transformed back to actual parameter space
using a pram config

options are
    min
    max
    param_type | float, int, categorical
    if categorical -> values: [] #list of values
    scale | linear or log
    grid: (int) whether the param should be snapped to a grid, value indicates no. elements
"""
from __future__ import annotations

from typing import Any, List, Literal, MutableMapping, Optional, Type, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from hola.utils import parse

JSON = Any
ParamName = str
TParamConfig = TypeVar("TParamConfig", bound="ParamConfig")
ParamConfigSpec = MutableMapping[ParamName, Any]
ParamsSpec = Union[
    MutableMapping[ParamName, "ParamConfig"], MutableMapping[ParamName, Any], MutableMapping[ParamName, str]
]


def parse_params_config(params_config: ParamsSpec) -> dict[ParamName, ParamConfig]:
    return {param_name: ParamConfig.parse(config) for param_name, config in params_config.items()}


class ParamConfig(BaseModel):
    """Parameter attributes."""

    min: float = 0
    max: float = 100
    scale: Literal["linear", "log"] = "linear"
    param_type: Literal["float", "int", "integer"] = "float"
    values: Optional[List[float]] = None
    grid: Optional[float] = None

    @property
    def resolved_param_type(self) -> Type[int | float]:
        if self.param_type in ["int", "integer"]:
            return int
        return float

    @classmethod
    def parse(cls: Type[TParamConfig], param_dict: TParamConfig | MutableMapping | str) -> TParamConfig:
        """Parse object, dictionary, or json string into an actual object."""
        return parse(cls, param_dict)

    def transform_param(self, u: float) -> int | float:
        """De-normalize parameter value from the [0, 1] interval."""
        if self.values is not None:
            n_values = len(self.values)
            for i in range(n_values):
                if u < (i + 1) * 1.0 / n_values:
                    a = self.values[i]
                    return a

        if self.scale == "log":
            a = np.log(self.min) + u * (np.log(self.max) - np.log(self.min))
            a = np.exp(a)
        else:
            a = self.min + u * (self.max - self.min)

        if self.grid is not None:
            if self.scale == "log":
                a = u * (np.log(self.max) - np.log(self.min))
                d = (np.log(self.max) - np.log(self.min)) / (self.grid - 1)
                a = int(a / d) * d + np.log(self.min)
                a = np.exp(a)
            else:
                d = (self.max - self.min) * 1.0 / (self.grid - 1)
                a = int(a / d) * d

        return self.resolved_param_type(a)

    def back_transform_param(self, x: float) -> float:
        """Normalize parameter value to the [0, 1] interval."""
        values, scale = self.values, self.scale

        if values is not None:
            for i, value in enumerate(values):
                if x == value:
                    high = float(i + 1) / len(values)
                    low = float(i) / len(values)
                    return low + np.random.rand() * (high - low)

        if scale == "log":
            a = np.log(x)
            a = (a - np.log(self.min)) / (np.log(self.max) - np.log(self.min))
        else:
            a = (x - self.min) / (self.max - self.min)
        return cast(float, a)


class ParameterTransformer:
    """Transform parameters to [0, 1] and back."""

    def __init__(self, params_config: dict[ParamName, ParamConfig]):
        self.params_config = params_config
        self.num_params = len(params_config)

    def back_transform_param_dict(self, param_dict: dict[ParamName, float]) -> npt.NDArray[np.floating]:
        u_params = np.zeros(self.num_params)
        for i, (param_name, param_val) in enumerate(param_dict.items()):
            param_config = self.params_config[param_name]
            u_params[i] = param_config.back_transform_param(param_val)
        return u_params

    def transform_u_params(self, u_param_sample: npt.NDArray[np.floating]) -> dict[ParamName, float]:
        param_dict: dict[ParamName, float] = {}
        for i, (param_name, param_config) in enumerate(self.params_config.items()):
            param_dict[param_name] = param_config.transform_param(u_param_sample[i])
        return param_dict
