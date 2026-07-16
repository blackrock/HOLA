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

import math
from collections.abc import Callable
from functools import partial

from benchmarks.functions import dtlz, wfg, zdt
from benchmarks.problems.registry import MultiObjectiveProblem


def _unit_bounds(n: int) -> dict[str, tuple[float, float]]:
    return {f"x{i}": (0.0, 1.0) for i in range(n)}


def _normalized_hypervolume_under_front(
    intervals: tuple[tuple[float, float], ...],
    front_value: Callable[[float], float],
    front_antiderivative: Callable[[float], float],
    ideal_point: tuple[float, float],
    reference_point: tuple[float, float],
) -> float:
    """Integrate a decreasing, possibly disconnected, two-objective front.

    Within a Pareto interval, the dominated height at abscissa ``x`` is
    ``reference_y - front_value(x)``.  Across a disconnected gap, it remains
    fixed at the preceding interval's right-end value.  The raw area is then
    divided by the fixed ideal-reference reporting box.
    """
    reference_x, reference_y = reference_point
    raw_hypervolume = 0.0
    for index, (left, right) in enumerate(intervals):
        raw_hypervolume += reference_y * (right - left)
        raw_hypervolume -= front_antiderivative(right) - front_antiderivative(left)
        next_left = intervals[index + 1][0] if index + 1 < len(intervals) else reference_x
        raw_hypervolume += (next_left - right) * (reference_y - front_value(right))

    ideal_x, ideal_y = ideal_point
    return raw_hypervolume / ((reference_x - ideal_x) * (reference_y - ideal_y))


def _zdt1_front_value(x: float) -> float:
    return 1.0 - math.sqrt(x)


def _zdt1_front_antiderivative(x: float) -> float:
    return x - (2.0 / 3.0) * x**1.5


def _zdt2_front_value(x: float) -> float:
    return 1.0 - x**2


def _zdt2_front_antiderivative(x: float) -> float:
    return x - x**3 / 3.0


def _zdt3_front_value(x: float) -> float:
    return 1.0 - math.sqrt(x) - x * math.sin(10.0 * math.pi * x)


def _zdt3_front_antiderivative(x: float) -> float:
    frequency = 10.0 * math.pi
    return (
        x
        - (2.0 / 3.0) * x**1.5
        + x * math.cos(frequency * x) / frequency
        - math.sin(frequency * x) / frequency**2
    )


def _dtlz1_normalized_hypervolume(n_obj: int, reference: float = 0.6) -> float:
    """Exact normalized HV outside the DTLZ1 simplex ``sum(f)=0.5``."""
    return 1.0 - (0.5 / reference) ** n_obj / math.factorial(n_obj)


def _dtlz_sphere_normalized_hypervolume(n_obj: int, reference: float = 1.1) -> float:
    """Exact normalized HV outside the positive-orthant unit sphere."""
    orthant_ball_volume = math.pi ** (n_obj / 2) / (2**n_obj * math.gamma(n_obj / 2 + 1))
    return 1.0 - orthant_ball_volume / reference**n_obj


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
        reference_point=(1.1, 1.1),
        ideal_point=(0.0, 0.0),
        family="zdt1",
        known_normalized_hypervolume=_normalized_hypervolume_under_front(
            ((0.0, 1.0),),
            _zdt1_front_value,
            _zdt1_front_antiderivative,
            (0.0, 0.0),
            (1.1, 1.1),
        ),
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
        reference_point=(1.1, 1.1),
        ideal_point=(0.0, 0.0),
        family="zdt2",
        known_normalized_hypervolume=_normalized_hypervolume_under_front(
            ((0.0, 1.0),),
            _zdt2_front_value,
            _zdt2_front_antiderivative,
            (0.0, 0.0),
            (1.1, 1.1),
        ),
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
        reference_point=(1.1, 1.1),
        ideal_point=(0.0, zdt.ZDT3_IDEAL_F2),
        family="zdt3",
        known_normalized_hypervolume=_normalized_hypervolume_under_front(
            zdt.ZDT3_PARETO_INTERVALS,
            _zdt3_front_value,
            _zdt3_front_antiderivative,
            (0.0, zdt.ZDT3_IDEAL_F2),
            (1.1, 1.1),
        ),
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
        reference_point=(1.1, 1.1),
        ideal_point=(0.0, 0.0),
        family="zdt4",
        known_normalized_hypervolume=_normalized_hypervolume_under_front(
            ((0.0, 1.0),),
            _zdt1_front_value,
            _zdt1_front_antiderivative,
            (0.0, 0.0),
            (1.1, 1.1),
        ),
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
        reference_point=(1.1, 1.1),
        ideal_point=(zdt.ZDT6_IDEAL_F1, 0.0),
        family="zdt6",
        known_normalized_hypervolume=_normalized_hypervolume_under_front(
            ((zdt.ZDT6_IDEAL_F1, 1.0),),
            _zdt2_front_value,
            _zdt2_front_antiderivative,
            (zdt.ZDT6_IDEAL_F1, 0.0),
            (1.1, 1.1),
        ),
        true_pareto_front=zdt.zdt6_pareto_front(),
        description="Non-uniform, biased",
    )
)

# ---------------------------------------------------------------------------
# DTLZ family (3 and 5 objectives)
# ---------------------------------------------------------------------------

for n_obj in (3, 5):
    # DTLZ1: k = n - M + 1 = 5, so n = M + 4.
    n_vars = n_obj + 4
    _register(
        MultiObjectiveProblem(
            name=f"dtlz1_{n_obj}obj_{n_vars}d",
            func=partial(dtlz.dtlz1, n_obj=n_obj),
            bounds=_unit_bounds(n_vars),
            objective_names=tuple(f"f{i + 1}" for i in range(n_obj)),
            reference_point=tuple(0.6 for _ in range(n_obj)),
            ideal_point=tuple(0.0 for _ in range(n_obj)),
            family="dtlz1",
            known_normalized_hypervolume=_dtlz1_normalized_hypervolume(n_obj),
            true_pareto_front=dtlz.dtlz1_pareto_front(
                n_obj,
                n_points=4096 if n_obj == 3 else 2048,
            ),
            description=f"Linear hyperplane, {n_obj} objectives",
        )
    )

    # DTLZ2, 3, 4: k = n - M + 1 = 10, so n = M + 9.
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
                reference_point=tuple(1.1 for _ in range(n_obj)),
                ideal_point=tuple(0.0 for _ in range(n_obj)),
                family=variant,
                known_normalized_hypervolume=_dtlz_sphere_normalized_hypervolume(n_obj),
                true_pareto_front=dtlz.dtlz2_pareto_front(
                    n_obj,
                    n_points=4096 if n_obj == 3 else 2048,
                ),
                description=f"{desc}, {n_obj} objectives",
            )
        )

# ---------------------------------------------------------------------------
# WFG family (2 and 3 objectives)
# ---------------------------------------------------------------------------

for n_obj in (2, 3):
    # Huband et al.'s recommended WFG construction: k=4 for the configured
    # two- and three-objective problems, with l=20 distance parameters.
    k = 4 if n_obj == 2 else 2 * (n_obj - 1)
    n_dist = 20
    n_vars = k + n_dist

    for variant, fn, pf_fn, desc in [
        ("wfg1", wfg.wfg1, wfg.wfg1_pareto_front, "Mixed convex/concave, biased"),
        ("wfg4", wfg.wfg4, wfg.wfg4_pareto_front, "Multimodal"),
        ("wfg9", wfg.wfg9, wfg.wfg9_pareto_front, "Non-separable, deceptive"),
    ]:
        true_pareto_front = pf_fn(n_obj)
        # At WFG1's nominal distance optimum, pymoo's polynomial transform
        # amplifies a floating roundoff residual.  Normalize against the
        # coordinatewise minimum that its implemented evaluator can attain,
        # while retaining the fixed absolute reporting reference above it.
        ideal_point = (
            tuple(float(value) for value in true_pareto_front.min(axis=0))
            if variant == "wfg1"
            else tuple(0.0 for _ in range(n_obj))
        )
        _register(
            MultiObjectiveProblem(
                name=f"{variant}_{n_obj}obj_{n_vars}d",
                func=partial(fn, n_obj=n_obj),
                bounds=_unit_bounds(n_vars),
                objective_names=tuple(f"f{i + 1}" for i in range(n_obj)),
                reference_point=tuple(1.1 * 2.0 * (i + 1) for i in range(n_obj)),
                ideal_point=ideal_point,
                family=variant,
                true_pareto_front=true_pareto_front,
                description=f"{desc}, {n_obj} objectives",
            )
        )
