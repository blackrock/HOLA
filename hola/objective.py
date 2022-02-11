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
from typing import Dict, MutableMapping, Type, TypeVar, Union

import numpy as np
from pydantic import BaseModel

from hola.utils import parse

FuncName = str
TObjectiveConfig = TypeVar("TObjectiveConfig", bound="ObjectiveConfig")
ObjectiveConfigSpec = Dict[str, float]
ObjectivesSpec = Union[MutableMapping[FuncName, "ObjectiveConfig"], MutableMapping[FuncName, ObjectiveConfigSpec]]


def parse_objectives_config(obj_config: ObjectivesSpec) -> dict[FuncName, ObjectiveConfig]:
    return {func_name: ObjectiveConfig.parse(config) for func_name, config in obj_config.items()}


class ObjectiveConfig(BaseModel):
    """Configuration of Target-Priority-Limit scalarizition for objectives."""

    target: float
    limit: float
    priority: float = 1.0

    @classmethod
    def parse(
        cls: Type[TObjectiveConfig], obj_config_dict: TObjectiveConfig | MutableMapping | str
    ) -> TObjectiveConfig:
        """Parse object, dictionary or json string into object."""
        return parse(cls, obj_config_dict)


@dataclass
class ObjectiveScalarizer:
    """Target-Priority-Limit scalarizition."""

    objectives_config: dict[FuncName, ObjectiveConfig]

    def scalarize_objectives(self, objectives_dict: dict[FuncName, float]) -> float:
        scalarized_objs: float = 0
        for obj_name, value in objectives_dict.items():
            obj_config = self.objectives_config[obj_name]
            scalarized_objs += scalarize(value, obj_config)
        return scalarized_objs


def scalarize(value: float, config: ObjectiveConfig) -> float:
    target, limit, priority = config.target, config.limit, config.priority
    if config.limit < target:  # pylint: disable=no-else-return
        if value < limit:
            return np.inf
        if value > target:
            return 0.0
        return priority - priority * (value - limit) / (target - limit)
    else:
        if value < target:
            return 0.0
        if value > limit:
            return np.inf
        return priority * (value - target) / (limit - target)
