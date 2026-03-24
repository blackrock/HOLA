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

"""Protocol classes and result types for optimizer adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from benchmarks.problems.registry import (
    MultiObjectiveProblem,
    SingleObjectiveProblem,
)


@dataclass
class SingleObjectiveResult:
    """Result of a single-objective optimization run."""

    best_value: float
    best_params: dict[str, float]
    wall_time_seconds: float
    convergence_trace: list[float] = field(default_factory=list)


@dataclass
class MultiObjectiveResult:
    """Result of a multi-objective optimization run."""

    pareto_front: np.ndarray  # (N, M) array of non-dominated objective values
    wall_time_seconds: float


@runtime_checkable
class SingleObjectiveOptimizer(Protocol):
    """Interface for single-objective optimizers."""

    name: str

    def optimize(
        self,
        problem: SingleObjectiveProblem,
        budget: int,
        seed: int,
    ) -> SingleObjectiveResult: ...


@runtime_checkable
class MultiObjectiveOptimizer(Protocol):
    """Interface for multi-objective optimizers."""

    name: str

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        budget: int,
        seed: int,
    ) -> MultiObjectiveResult: ...
