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

"""Dedicated mixed-space hyperparameter-optimization problem definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal, Protocol, TypeAlias


class HpoEvaluator(Protocol):
    """One run-scoped validation/test split with explicit call accounting."""

    validation_budget: int
    validation_calls: int
    heldout_calls: int

    @property
    def split_sizes(self) -> tuple[int, int, int]: ...

    def evaluate_validation(self, params: dict[str, Any]) -> float: ...

    def evaluate_heldout(self, params: dict[str, Any]) -> float: ...


@dataclass(frozen=True)
class IntegerParameter:
    """Inclusive native integer parameter."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("integer parameter minimum must not exceed maximum")

    def normalize(self, name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"HPO parameter {name!r} must be an integer")
        normalized = int(value)
        if not self.minimum <= normalized <= self.maximum:
            raise ValueError(f"HPO parameter {name!r} is outside its registered range")
        return normalized

    def configuration(self) -> dict[str, object]:
        return {"kind": "integer", "minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True)
class RealParameter:
    """Native real parameter with a fixed linear or log10 sampling scale."""

    minimum: float
    maximum: float
    scale: Literal["linear", "log10"] = "linear"

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("real parameter minimum must not exceed maximum")
        if self.scale == "log10" and self.minimum <= 0.0:
            raise ValueError("log10 real parameters require a positive minimum")

    def normalize(self, name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"HPO parameter {name!r} must be real-valued")
        normalized = float(value)
        if not self.minimum <= normalized <= self.maximum:
            raise ValueError(f"HPO parameter {name!r} is outside its registered range")
        return normalized

    def configuration(self) -> dict[str, object]:
        return {
            "kind": "real",
            "minimum": self.minimum,
            "maximum": self.maximum,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class CategoricalParameter:
    """Native categorical parameter with string-valued choices."""

    choices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("categorical parameters require at least one choice")
        if len(self.choices) != len(set(self.choices)):
            raise ValueError("categorical parameter choices must be unique")

    def normalize(self, name: str, value: Any) -> str:
        if not isinstance(value, str) or value not in self.choices:
            raise ValueError(f"HPO parameter {name!r} is not a registered category")
        return value

    def configuration(self) -> dict[str, object]:
        return {"kind": "categorical", "choices": list(self.choices)}


HpoParameter: TypeAlias = IntegerParameter | RealParameter | CategoricalParameter
EvaluatorFactory: TypeAlias = Callable[["HpoProblem", int, int], HpoEvaluator]


@dataclass(frozen=True)
class HpoProblem:
    """A practical HPO task kept separate from all-real analytic functions."""

    name: str
    parameters: Mapping[str, HpoParameter]
    evaluator_factory: EvaluatorFactory = field(repr=False)
    objective_name: str = "validation_r2"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HPO problem name must be non-empty")
        if not self.parameters:
            raise ValueError(f"{self.name} must define at least one parameter")

    def normalize_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        missing = set(self.parameters) - set(params)
        extra = set(params) - set(self.parameters)
        if missing or extra:
            raise ValueError(
                f"{self.name} parameter mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return {
            name: specification.normalize(name, params[name])
            for name, specification in self.parameters.items()
        }

    def make_evaluator(self, split_seed: int, validation_budget: int) -> HpoEvaluator:
        """Create one evaluator from the run-scoped, horizon-independent split seed."""
        if validation_budget <= 0:
            raise ValueError("HPO validation budget must be positive")
        return self.evaluator_factory(self, split_seed, validation_budget)

    def configuration(self) -> dict[str, object]:
        return {
            "name": self.name,
            "objective": self.objective_name,
            "direction": "maximize",
            "parameters": {
                name: specification.configuration()
                for name, specification in self.parameters.items()
            },
            "split": {
                "dataset": "sklearn.datasets.load_diabetes",
                "test_fraction": 0.20,
                "validation_fraction_of_train_validation": 0.25,
                "paired_seeded_split": True,
                "split_seed_scope": "problem+run_id; fixed across budgets and optimizers",
                "optimization_access": "training and validation only",
                "heldout_access": "one final refit-and-score after selection",
            },
        }


def _make_diabetes_evaluator(
    problem: HpoProblem,
    split_seed: int,
    validation_budget: int,
) -> HpoEvaluator:
    from benchmarks.functions.hpo import DiabetesGbrEvaluator

    return DiabetesGbrEvaluator(problem, split_seed, validation_budget)


DIABETES_GBR_HPO = HpoProblem(
    name="gbr_diabetes_hpo",
    parameters={
        "n_estimators": IntegerParameter(50, 500),
        "max_depth": IntegerParameter(1, 6),
        "learning_rate": RealParameter(1e-3, 0.3, scale="log10"),
        "subsample": RealParameter(0.5, 1.0),
        "loss": CategoricalParameter(("squared_error", "huber", "absolute_error")),
    },
    evaluator_factory=_make_diabetes_evaluator,
    description="Gradient-boosted regression on a fixed diabetes train/validation/test split",
)

HPO_PROBLEMS: dict[str, HpoProblem] = {DIABETES_GBR_HPO.name: DIABETES_GBR_HPO}
