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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from benchmarks.problems.hpo import HpoProblem
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
    n_evaluations: int
    convergence_trace: list[float] = field(default_factory=list)


@dataclass
class MultiObjectiveResult:
    """Result of a multi-objective optimization run."""

    pareto_front: np.ndarray  # (N, M) array of non-dominated objective values
    wall_time_seconds: float
    n_evaluations: int
    decision_vectors: np.ndarray | None = None


@dataclass
class HpoOptimizationResult:
    """Validation-only optimizer result; held-out scoring belongs to the runner."""

    best_validation_value: float
    best_params: dict[str, Any]
    wall_time_seconds: float
    n_evaluations: int
    validation_trace: list[float] = field(default_factory=list)


class EvaluationCountError(RuntimeError):
    """An optimizer violated its declared objective-evaluation contract."""

    def __init__(self, actual: int, expected: int, optimizer: str) -> None:
        self.actual = actual
        self.expected = expected
        self.optimizer = optimizer
        super().__init__(
            f"{optimizer} used {actual} objective evaluations; expected exactly {expected}"
        )


def assert_exact_evaluations(actual: int, expected: int, optimizer: str) -> None:
    """Fail a benchmark run whose objective-call count missed its contract."""
    if actual != expected:
        raise EvaluationCountError(actual, expected, optimizer)


def optimizer_configuration(optimizer: object, budget: int) -> dict[str, Any]:
    """Return the adapter's resolved, JSON-serializable run configuration."""
    configuration = getattr(optimizer, "configuration", None)
    if configuration is None:
        return {"adapter": type(optimizer).__name__, "name": getattr(optimizer, "name", "")}
    value = configuration(budget)
    if not isinstance(value, dict):
        raise TypeError(f"{type(optimizer).__name__}.configuration() must return a dict")
    return value


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


@runtime_checkable
class HpoOptimizer(Protocol):
    """Interface for mixed-space validation-only HPO adapters."""

    name: str

    def optimize(
        self,
        problem: HpoProblem,
        evaluate_validation: Callable[[dict[str, Any]], float],
        budget: int,
        seed: int,
    ) -> HpoOptimizationResult: ...
