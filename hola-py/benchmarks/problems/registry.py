# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Problem dataclasses and central registries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SingleObjectiveProblem:
    """A single-objective benchmark problem."""

    name: str
    func: Callable[[dict[str, float]], float]
    bounds: dict[str, tuple[float, float]]
    known_minimum: float
    description: str = ""

    @property
    def dimensionality(self) -> int:
        return len(self.bounds)


@dataclass(frozen=True)
class MultiObjectiveProblem:
    """A multi-objective benchmark problem."""

    name: str
    func: Callable[[dict[str, float]], dict[str, float]]
    bounds: dict[str, tuple[float, float]]
    objective_names: tuple[str, ...]
    reference_point: tuple[float, ...]
    true_pareto_front: np.ndarray | None = field(default=None, repr=False)
    description: str = ""

    @property
    def n_objectives(self) -> int:
        return len(self.objective_names)

    @property
    def dimensionality(self) -> int:
        return len(self.bounds)


@dataclass(frozen=True)
class GroupedTlpObjective:
    """Configuration for one objective in a grouped TLP problem."""

    field: str
    sense: str  # "minimize" or "maximize"
    target: float
    limit: float
    priority: float  # objectives with same priority form a group


@dataclass(frozen=True)
class GroupedTlpProblem:
    """A grouped TLP benchmark problem (HOLA-specific)."""

    name: str
    func: Callable[[dict[str, float]], dict[str, float]]
    bounds: dict[str, tuple[float, float]]
    objectives: tuple[GroupedTlpObjective, ...]
    reference_point: tuple[float, ...]  # in group-cost space
    n_groups: int
    true_pareto_front: np.ndarray | None = field(default=None, repr=False)
    description: str = ""

    @property
    def dimensionality(self) -> int:
        return len(self.bounds)
