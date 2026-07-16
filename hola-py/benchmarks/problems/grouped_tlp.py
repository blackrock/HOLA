# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicitly grouped TLP capability problem."""

from __future__ import annotations

from benchmarks.functions.grouped_tlp import (
    synthetic_grouped_tlp,
    synthetic_grouped_tlp_pareto_front,
)
from benchmarks.problems.registry import GroupedTlpObjective, GroupedTlpProblem

_LINEAR_TARGET = 0.2
_LINEAR_LIMIT = 0.9
_QUADRATIC_TARGET = _LINEAR_TARGET**2
_QUADRATIC_LIMIT = _LINEAR_LIMIT**2

SYNTHETIC_GROUPED_TLP = GroupedTlpProblem(
    name="synthetic_grouped_tlp_5d",
    raw_func=synthetic_grouped_tlp,
    bounds={f"x{index}": (0.0, 1.0) for index in range(5)},
    objectives=(
        GroupedTlpObjective(
            "f1",
            "minimize",
            target=_LINEAR_TARGET,
            limit=_LINEAR_LIMIT,
            priority=1.0,
            group="group_a",
        ),
        GroupedTlpObjective(
            "f2",
            "minimize",
            target=_QUADRATIC_TARGET,
            limit=_QUADRATIC_LIMIT,
            priority=2.0,
            group="group_a",
        ),
        GroupedTlpObjective(
            "f3",
            "minimize",
            target=_LINEAR_TARGET,
            limit=_LINEAR_LIMIT,
            priority=2.0,
            group="group_b",
        ),
        GroupedTlpObjective(
            "f4",
            "minimize",
            target=_QUADRATIC_TARGET,
            limit=_QUADRATIC_LIMIT,
            priority=1.0,
            group="group_b",
        ),
    ),
    ideal_point=(0.0, 0.0),
    reference_point=(3.0, 3.0),
    true_pareto_front=synthetic_grouped_tlp_pareto_front(),
    # Exact unit-box hypervolume: 1 - (1/9) * integral B(x) A'(x) dx
    # from x=1/5 to x=4/5, where A and B are the two group costs.
    known_normalized_hypervolume=4325.0 / 5929.0,
    description=(
        "Four raw objectives with target plateaus, reachable limit violations, "
        "two explicit TLP groups, and unequal within-group priorities"
    ),
)

GROUPED_TLP_PROBLEMS: dict[str, GroupedTlpProblem] = {
    SYNTHETIC_GROUPED_TLP.name: SYNTHETIC_GROUPED_TLP
}
