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

"""Grouped TLP problem instances."""

from __future__ import annotations

from benchmarks.functions.grouped_tlp import ml_grouped, synthetic_grouped
from benchmarks.problems.registry import GroupedTlpObjective, GroupedTlpProblem


def _unit_bounds(n: int) -> dict[str, tuple[float, float]]:
    return {f"x{i}": (0.0, 1.0) for i in range(n)}


GROUPED_TLP_PROBLEMS: dict[str, GroupedTlpProblem] = {}


def _register(p: GroupedTlpProblem) -> None:
    GROUPED_TLP_PROBLEMS[p.name] = p


# Synthetic: 4 objectives in 2 groups of 2
_register(
    GroupedTlpProblem(
        name="synthetic_grouped_5d",
        func=synthetic_grouped,
        bounds=_unit_bounds(5),
        objectives=(
            GroupedTlpObjective("f1", "minimize", target=0.0, limit=5.0, priority=1.0),
            GroupedTlpObjective("f2", "minimize", target=0.0, limit=5.0, priority=1.0),
            GroupedTlpObjective("f3", "minimize", target=0.0, limit=5.0, priority=2.0),
            GroupedTlpObjective("f4", "minimize", target=0.0, limit=5.0, priority=2.0),
        ),
        reference_point=(3.0, 3.0),
        n_groups=2,
        description="Synthetic: 4 objectives, 2 groups of 2, known Pareto front",
    )
)

# ML-inspired: performance group vs resource group
_register(
    GroupedTlpProblem(
        name="ml_grouped_gbr",
        func=ml_grouped,
        bounds={
            "n_estimators": (10.0, 500.0),
            "max_depth": (1.0, 6.0),
            "learning_rate": (1e-4, 1.0),
            "subsample": (0.2, 1.0),
        },
        objectives=(
            # Performance group (priority=1): maximize R^2 and F1
            GroupedTlpObjective("r2", "maximize", target=0.95, limit=0.0, priority=1.0),
            GroupedTlpObjective("f1", "maximize", target=0.90, limit=0.3, priority=1.0),
            # Resource group (priority=2): minimize time and model size
            GroupedTlpObjective("train_time", "minimize", target=0.1, limit=10.0, priority=2.0),
            GroupedTlpObjective("model_size", "minimize", target=100.0, limit=3000.0, priority=2.0),
        ),
        reference_point=(3.0, 3.0),
        n_groups=2,
        description="ML: performance (R^2, F1) vs resources (time, size)",
    )
)
