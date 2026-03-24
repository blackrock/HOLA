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

"""Multi-objective problem instances (ZDT, DTLZ, WFG)."""

from __future__ import annotations

from functools import partial

from benchmarks.functions import dtlz, wfg, zdt
from benchmarks.problems.registry import MultiObjectiveProblem


def _unit_bounds(n: int) -> dict[str, tuple[float, float]]:
    return {f"x{i}": (0.0, 1.0) for i in range(n)}


MULTI_OBJECTIVE_PROBLEMS: dict[str, MultiObjectiveProblem] = {}


def _register(p: MultiObjectiveProblem) -> None:
    MULTI_OBJECTIVE_PROBLEMS[p.name] = p


# ---------------------------------------------------------------------------
# ZDT family (2 objectives)
# ---------------------------------------------------------------------------

_register(
    MultiObjectiveProblem(
        name="zdt1_30d",
        func=zdt.zdt1,
        bounds=_unit_bounds(30),
        objective_names=("f1", "f2"),
        reference_point=(11.0, 11.0),
        true_pareto_front=zdt.zdt1_pareto_front(),
        description="Convex Pareto front",
    )
)

_register(
    MultiObjectiveProblem(
        name="zdt2_30d",
        func=zdt.zdt2,
        bounds=_unit_bounds(30),
        objective_names=("f1", "f2"),
        reference_point=(11.0, 11.0),
        true_pareto_front=zdt.zdt2_pareto_front(),
        description="Non-convex (concave) Pareto front",
    )
)

_register(
    MultiObjectiveProblem(
        name="zdt3_30d",
        func=zdt.zdt3,
        bounds=_unit_bounds(30),
        objective_names=("f1", "f2"),
        reference_point=(11.0, 11.0),
        true_pareto_front=zdt.zdt3_pareto_front(),
        description="Disconnected Pareto front",
    )
)

_register(
    MultiObjectiveProblem(
        name="zdt4_10d",
        func=zdt.zdt4,
        bounds={
            "x0": (0.0, 1.0),
            **{f"x{i}": (-5.0, 5.0) for i in range(1, 10)},
        },
        objective_names=("f1", "f2"),
        reference_point=(11.0, 11.0),
        true_pareto_front=zdt.zdt4_pareto_front(),
        description="Multimodal, many local fronts",
    )
)

_register(
    MultiObjectiveProblem(
        name="zdt6_10d",
        func=zdt.zdt6,
        bounds=_unit_bounds(10),
        objective_names=("f1", "f2"),
        reference_point=(11.0, 11.0),
        true_pareto_front=zdt.zdt6_pareto_front(),
        description="Non-uniform, biased",
    )
)

# ---------------------------------------------------------------------------
# DTLZ family (3 and 5 objectives)
# ---------------------------------------------------------------------------

for n_obj in (3, 5):
    # DTLZ1: k=4, so n = M + 4
    n_vars = n_obj + 4
    _register(
        MultiObjectiveProblem(
            name=f"dtlz1_{n_obj}obj_{n_vars}d",
            func=partial(dtlz.dtlz1, n_obj=n_obj),
            bounds=_unit_bounds(n_vars),
            objective_names=tuple(f"f{i + 1}" for i in range(n_obj)),
            reference_point=tuple(1.0 for _ in range(n_obj)),
            true_pareto_front=dtlz.dtlz1_pareto_front(n_obj),
            description=f"Linear hyperplane, {n_obj} objectives",
        )
    )

    # DTLZ2, 3, 4: k=9, so n = M + 9
    n_vars = n_obj + 9
    for variant, fn, desc in [
        ("dtlz2", dtlz.dtlz2, "Spherical"),
        ("dtlz3", dtlz.dtlz3, "Spherical, multimodal"),
        ("dtlz4", dtlz.dtlz4, "Spherical, biased density"),
    ]:
        _register(
            MultiObjectiveProblem(
                name=f"{variant}_{n_obj}obj_{n_vars}d",
                func=partial(fn, n_obj=n_obj),
                bounds=_unit_bounds(n_vars),
                objective_names=tuple(f"f{i + 1}" for i in range(n_obj)),
                reference_point=tuple(2.0 for _ in range(n_obj)),
                true_pareto_front=dtlz.dtlz2_pareto_front(n_obj),
                description=f"{desc}, {n_obj} objectives",
            )
        )

# ---------------------------------------------------------------------------
# WFG family (2 and 3 objectives)
# ---------------------------------------------------------------------------

for n_obj in (2, 3):
    # WFG standard: k = n_obj - 1 position params, l = 4 distance params
    k = n_obj - 1
    n_dist = 4
    n_vars = k + n_dist

    for variant, fn, desc in [
        ("wfg1", wfg.wfg1, "Mixed convex/concave, biased"),
        ("wfg4", wfg.wfg4, "Multimodal"),
        ("wfg9", wfg.wfg9, "Non-separable, deceptive"),
    ]:
        _register(
            MultiObjectiveProblem(
                name=f"{variant}_{n_obj}obj_{n_vars}d",
                func=partial(fn, n_obj=n_obj),
                bounds=_unit_bounds(n_vars),
                objective_names=tuple(f"f{i + 1}" for i in range(n_obj)),
                reference_point=tuple(2.0 * (i + 1) + 1 for i in range(n_obj)),
                description=f"{desc}, {n_obj} objectives",
            )
        )
