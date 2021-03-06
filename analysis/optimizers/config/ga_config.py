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
from dataclasses import dataclass
from typing import Type, TypeVar

TGAConfig = TypeVar("TGAConfig", bound="GAConfig")


@dataclass
class GAConfig:
    pop_size: int
    configuration: int

    @classmethod
    def build(cls: Type[TGAConfig], config: int) -> TGAConfig:
        if config == 1:
            return cls(pop_size=5, configuration=config)
        if config == 2:
            return cls(pop_size=10, configuration=config)
        if config == 3:
            return cls(pop_size=25, configuration=config)
        raise ValueError(f"Invalid config number for {cls}")
