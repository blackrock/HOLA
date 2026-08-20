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

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

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


class EmpiricalExploitationError(RuntimeError):
    """A GMM benchmark run did not produce authenticated empirical exploitation."""

    def __init__(self, actual: int, observed_diagnostics: Mapping[str, object]) -> None:
        self.actual = actual
        self.observed_diagnostics = dict(observed_diagnostics)
        observed = json.dumps(
            self.observed_diagnostics,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        super().__init__(
            "GMM empirical-exploitation gate failed "
            "(requires gmm_fit_epoch>=1, gmm_sampling_ready=true, "
            "gmm_origin_suggestions>=5, and "
            "issued_suggestions==completed_evaluations); "
            f"observed_diagnostics={observed}"
        )


def empirical_exploitation_gate_configuration() -> dict[str, object]:
    """Return the manifest-bound practical-benchmark GMM gate."""

    return {
        "minimum_gmm_fit_epoch": 1,
        "minimum_gmm_origin_suggestions": 5,
        "on_failure": "preserved_error_outcome",
    }


def _stable_observed_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return value if np.isfinite(value) else f"<nonfinite {value!r}>"
    return f"<malformed {type(value).__name__}>"


def require_empirical_gmm_exploitation(study: object, completed_evaluations: int) -> None:
    """Fail closed unless a completed GMM run demonstrably used its fitted sampler."""

    if type(completed_evaluations) is not int or completed_evaluations < 0:
        raise ValueError("completed_evaluations must be a non-negative integer")
    required_fields = (
        "gmm_fit_epoch",
        "gmm_origin_suggestions",
        "gmm_sampling_ready",
        "issued_suggestions",
    )
    observed: dict[str, object] = {"completed_evaluations": completed_evaluations}
    try:
        diagnostics_method = cast(Any, study).strategy_diagnostics
        diagnostics = diagnostics_method()
    except Exception as error:
        observed["diagnostics_error"] = f"{type(error).__name__}: {error}"
        raise EmpiricalExploitationError(completed_evaluations, observed) from error
    if type(diagnostics) is not dict:
        observed["diagnostics"] = f"<malformed {type(diagnostics).__name__}>"
        raise EmpiricalExploitationError(completed_evaluations, observed)
    diagnostics = cast(dict[object, object], diagnostics)
    for field_name in required_fields:
        observed[field_name] = (
            "<missing>"
            if field_name not in diagnostics
            else _stable_observed_value(diagnostics[field_name])
        )

    fit_epoch = diagnostics.get("gmm_fit_epoch")
    origin_suggestions = diagnostics.get("gmm_origin_suggestions")
    sampling_ready = diagnostics.get("gmm_sampling_ready")
    issued_suggestions = diagnostics.get("issued_suggestions")
    valid = (
        type(fit_epoch) is int
        and fit_epoch >= 1
        and type(origin_suggestions) is int
        and origin_suggestions >= 5
        and sampling_ready is True
        and type(issued_suggestions) is int
        and issued_suggestions == completed_evaluations
    )
    if not valid:
        raise EmpiricalExploitationError(completed_evaluations, observed)


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
