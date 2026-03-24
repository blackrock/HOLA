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

"""Random search x2 adapter (double budget baseline)."""

from __future__ import annotations

from benchmarks.adapters.base import SingleObjectiveResult
from benchmarks.adapters.hola_adapter import HolaSingleObjectiveAdapter
from benchmarks.problems.registry import SingleObjectiveProblem


class RandomDoubleAdapter:
    """Random search with twice the iteration budget.

    Delegates to HOLA's random strategy but runs 2 * budget trials.
    This is a calibration baseline: if HOLA cannot beat random search
    with double the budget, it is not providing value.
    """

    name = "Random x2"

    def __init__(self) -> None:
        self._inner = HolaSingleObjectiveAdapter(strategy="random")

    def optimize(
        self,
        problem: SingleObjectiveProblem,
        budget: int,
        seed: int,
    ) -> SingleObjectiveResult:
        return self._inner.optimize(problem, budget * 2, seed)
