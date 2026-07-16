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
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class SingleObjectiveProblem:
    """A single-objective benchmark problem."""

    name: str
    func: Callable[[dict[str, float]], float]
    bounds: dict[str, tuple[float, float]]
    known_minimum: float
    family: str = ""
    suite: str = "synthetic"
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
    ideal_point: tuple[float, ...] | None = None
    family: str = ""
    known_normalized_hypervolume: float | None = None
    true_pareto_front: np.ndarray | None = field(default=None, repr=False)
    description: str = ""
    infeasible_sentinel: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.ideal_point is not None:
            scales = np.asarray(self.objective_scale)
            if self.true_pareto_front is not None:
                front = np.asarray(self.true_pareto_front, dtype=float)
                if front.ndim != 2 or front.shape[1] != self.n_objectives:
                    raise ValueError(f"{self.name} true front has the wrong shape")
                if front.size == 0 or not np.all(np.isfinite(front)):
                    raise ValueError(f"{self.name} true front must contain finite points")
                ideal = np.asarray(self.ideal_point)
                normalized = (front - ideal) / scales
                tolerance = 1e-12
                if np.any(normalized < -tolerance):
                    raise ValueError(f"{self.name} true front falls below its documented ideal")
                if np.any(normalized >= 1.0):
                    raise ValueError(
                        f"{self.name} true front must lie strictly inside its reference point"
                    )
        if self.known_normalized_hypervolume is not None and not (
            0.0 < self.known_normalized_hypervolume <= 1.0
        ):
            raise ValueError(f"{self.name} known normalized hypervolume must be in (0, 1]")
        if self.infeasible_sentinel is not None:
            if len(self.infeasible_sentinel) != self.n_objectives or not all(
                np.isfinite(value) for value in self.infeasible_sentinel
            ):
                raise ValueError(f"{self.name} has an invalid infeasible sentinel")
            if any(
                sentinel <= reference
                for sentinel, reference in zip(
                    self.infeasible_sentinel,
                    self.reference_point,
                    strict=True,
                )
            ):
                raise ValueError(
                    f"{self.name} infeasible sentinel must lie beyond its reference point"
                )

    @property
    def n_objectives(self) -> int:
        return len(self.objective_names)

    @property
    def dimensionality(self) -> int:
        return len(self.bounds)

    def is_infeasible_objectives(self, values: object) -> bool:
        """Return whether a library-facing objective vector is the sentinel."""
        if self.infeasible_sentinel is None:
            return False
        array = np.asarray(values, dtype=float)
        return bool(
            array.shape == (self.n_objectives,)
            and np.array_equal(array, np.asarray(self.infeasible_sentinel))
        )

    @property
    def objective_scale(self) -> tuple[float, ...]:
        """Fixed coordinate scales from the ideal to reporting reference point."""
        if self.ideal_point is None:
            raise ValueError(f"{self.name} does not define an objective ideal point")
        if len(self.ideal_point) != self.n_objectives:
            raise ValueError(f"{self.name} ideal point has the wrong dimension")
        if len(self.reference_point) != self.n_objectives:
            raise ValueError(f"{self.name} reference point has the wrong dimension")
        scales = tuple(
            reference - ideal
            for ideal, reference in zip(self.ideal_point, self.reference_point, strict=True)
        )
        if any(scale <= 0.0 for scale in scales):
            raise ValueError(f"{self.name} objective scales must all be positive")
        return scales

    @cached_property
    def normalized_reference_hypervolume(self) -> float:
        """Known or once-computed normalized hypervolume of the true front."""
        if self.known_normalized_hypervolume is not None:
            return self.known_normalized_hypervolume
        if self.ideal_point is None or self.true_pareto_front is None:
            raise ValueError(f"{self.name} lacks fixed hypervolume reference geometry")

        from benchmarks.metrics.hypervolume import compute_hv
        from benchmarks.metrics.normalization import normalize_objectives

        normalized = normalize_objectives(
            self.true_pareto_front,
            self.ideal_point,
            self.reference_point,
        )
        return compute_hv(normalized, np.ones(self.n_objectives))


@dataclass(frozen=True)
class GroupedTlpObjective:
    """One raw objective in an explicitly grouped TLP benchmark."""

    field: str
    sense: str
    target: float
    limit: float
    priority: float
    group: str

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("grouped TLP objective fields must be non-empty")
        if not self.group:
            raise ValueError(f"grouped TLP objective {self.field!r} needs an explicit group")
        if self.sense not in {"minimize", "maximize"}:
            raise ValueError(f"unsupported objective sense {self.sense!r}")
        if not all(np.isfinite(value) for value in (self.target, self.limit, self.priority)):
            raise ValueError(f"grouped TLP objective {self.field!r} must use finite settings")
        if self.priority < 0.0:
            raise ValueError(f"grouped TLP objective {self.field!r} has negative priority")
        if self.sense == "minimize" and self.target >= self.limit:
            raise ValueError(f"minimization objective {self.field!r} requires target < limit")
        if self.sense == "maximize" and self.target <= self.limit:
            raise ValueError(f"maximization objective {self.field!r} requires target > limit")

    def normalized_cost(self, value: float) -> float:
        """Apply target-limit normalization without the within-group priority weight."""
        if self.sense == "minimize":
            if value <= self.target:
                return 0.0
            if value > self.limit:
                return float("inf")
            return (value - self.target) / (self.limit - self.target)

        if value >= self.target:
            return 0.0
        if value < self.limit:
            return float("inf")
        return (self.target - value) / (self.target - self.limit)


@dataclass(frozen=True)
class GroupedTlpProblem:
    """Raw objectives plus the explicit transform into Pareto-ranked group costs."""

    name: str
    raw_func: Callable[[dict[str, float]], dict[str, float]]
    bounds: dict[str, tuple[float, float]]
    objectives: tuple[GroupedTlpObjective, ...]
    ideal_point: tuple[float, ...]
    reference_point: tuple[float, ...]
    true_pareto_front: np.ndarray = field(repr=False)
    known_normalized_hypervolume: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError(f"{self.name} must define grouped TLP objectives")
        fields = [objective.field for objective in self.objectives]
        if len(fields) != len(set(fields)):
            raise ValueError(f"{self.name} grouped TLP objective fields must be unique")
        if self.n_groups < 2:
            raise ValueError(f"{self.name} must define at least two explicit groups")
        # Reuse the ordinary multi-objective problem's fixed-geometry validation.
        self.as_multi_objective_problem()

    @property
    def group_names(self) -> tuple[str, ...]:
        """Stable group order defined by first appearance in the objective schema."""
        return tuple(dict.fromkeys(objective.group for objective in self.objectives))

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def dimensionality(self) -> int:
        return len(self.bounds)

    def group_costs(self, raw_metrics: dict[str, float]) -> dict[str, float]:
        """Sum priority-weighted normalized objective costs within each explicit group."""
        costs = {group: 0.0 for group in self.group_names}
        for objective in self.objectives:
            if objective.field not in raw_metrics:
                raise KeyError(f"raw metrics are missing objective {objective.field!r}")
            normalized = objective.normalized_cost(float(raw_metrics[objective.field]))
            weighted = normalized if np.isinf(normalized) else objective.priority * normalized
            costs[objective.group] += weighted
        return costs

    def evaluate_group_costs(self, params: dict[str, float]) -> dict[str, float]:
        """Evaluate raw fields and retain each diagnostic group cost."""
        return self.group_costs(self.raw_func(params))

    def evaluate_competitor_costs(self, params: dict[str, float]) -> dict[str, float]:
        """Expose HOLA's global feasibility rule to ordinary MO optimizers.

        If any raw TLP objective exceeds its limit, HOLA treats the entire
        trial as infeasible rather than trading a finite group against an
        infinite group. Ordinary multi-objective libraries receive the same
        ordering by mapping such a trial to infinity on every group axis.
        """
        costs = self.evaluate_group_costs(params)
        if all(np.isfinite(cost) for cost in costs.values()):
            return costs
        return dict(zip(self.group_names, self.competitor_infeasible_sentinel, strict=True))

    @property
    def competitor_infeasible_sentinel(self) -> tuple[float, ...]:
        """Finite, globally dominated values safe for third-party MO arithmetic."""
        return tuple(reference + 1.0 for reference in self.reference_point)

    def as_multi_objective_problem(self) -> MultiObjectiveProblem:
        """Expose the group-cost problem through the shared multi-objective protocol."""
        return MultiObjectiveProblem(
            name=self.name,
            func=self.evaluate_competitor_costs,
            bounds=self.bounds,
            objective_names=self.group_names,
            ideal_point=self.ideal_point,
            reference_point=self.reference_point,
            family="grouped_tlp",
            known_normalized_hypervolume=self.known_normalized_hypervolume,
            true_pareto_front=self.true_pareto_front,
            description=self.description,
            infeasible_sentinel=self.competitor_infeasible_sentinel,
        )
