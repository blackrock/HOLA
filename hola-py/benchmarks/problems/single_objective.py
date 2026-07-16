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

"""Single-objective problem instances."""

from __future__ import annotations

from benchmarks.functions import single_objective as f
from benchmarks.problems.registry import SingleObjectiveProblem


def _nd_bounds(n: int, lo: float, hi: float) -> dict[str, tuple[float, float]]:
    return {f"x{i}": (lo, hi) for i in range(n)}


# ---------------------------------------------------------------------------
# Problem instances matching benchmark_design.md
# ---------------------------------------------------------------------------

SINGLE_OBJECTIVE_PROBLEMS: dict[str, SingleObjectiveProblem] = {}


def _register(p: SingleObjectiveProblem) -> None:
    SINGLE_OBJECTIVE_PROBLEMS[p.name] = p


# 1-D
_register(
    SingleObjectiveProblem(
        name="forrester_1d",
        func=f.forrester,
        bounds={"x": (0.0, 1.0)},
        known_minimum=-6.0207400557670825,
        family="forrester",
        description="1D, multiple local minima",
    )
)

# 2-D
_register(
    SingleObjectiveProblem(
        name="branin_2d",
        func=f.branin,
        bounds={"x1": (-5.0, 10.0), "x2": (0.0, 15.0)},
        known_minimum=0.39788735772973816,
        family="branin",
        description="2D, three global minima",
    )
)
_register(
    SingleObjectiveProblem(
        name="bukin6_2d",
        func=f.bukin_6,
        bounds={"x1": (-15.0, -5.0), "x2": (-3.0, 3.0)},
        known_minimum=0.0,
        family="bukin6",
        description="2D, narrow valley",
    )
)
_register(
    SingleObjectiveProblem(
        name="cross_in_tray_2d",
        func=f.cross_in_tray,
        bounds={"x1": (-10.0, 10.0), "x2": (-10.0, 10.0)},
        known_minimum=-2.062611870822739,
        family="cross_in_tray",
        description="2D, four symmetric minima",
    )
)
_register(
    SingleObjectiveProblem(
        name="drop_wave_2d",
        func=f.drop_wave,
        bounds={"x1": (-5.12, 5.12), "x2": (-5.12, 5.12)},
        known_minimum=-1.0,
        family="drop_wave",
        description="2D, many local minima",
    )
)
_register(
    SingleObjectiveProblem(
        name="egg_holder_2d",
        func=f.egg_holder,
        bounds={"x1": (-512.0, 512.0), "x2": (-512.0, 512.0)},
        known_minimum=-959.6406627208507,
        family="egg_holder",
        description="2D, deceptive landscape",
    )
)
_register(
    SingleObjectiveProblem(
        name="holder_table_2d",
        func=f.holder_table,
        bounds={"x1": (-10.0, 10.0), "x2": (-10.0, 10.0)},
        known_minimum=-19.20850256788675,
        family="holder_table",
        description="2D, four symmetric minima",
    )
)
_register(
    SingleObjectiveProblem(
        name="levy13_2d",
        func=f.levy_13,
        bounds={"x1": (-10.0, 10.0), "x2": (-10.0, 10.0)},
        known_minimum=0.0,
        family="levy13",
        description="2D, many local minima",
    )
)
_register(
    SingleObjectiveProblem(
        name="six_hump_camel_2d",
        func=f.six_hump_camel,
        bounds={"x1": (-3.0, 3.0), "x2": (-2.0, 2.0)},
        known_minimum=-1.031628453489877,
        family="six_hump_camel",
        description="2D, six local minima",
    )
)

# N-D: Ackley at 2, 5, 7 dimensions
for nd in (2, 5, 7):
    _register(
        SingleObjectiveProblem(
            name=f"ackley_{nd}d",
            func=f.ackley,
            bounds=_nd_bounds(nd, -32.768, 32.768),
            known_minimum=0.0,
            family="ackley",
            description=f"{nd}D, exponential + cosine, multimodal",
        )
    )

# N-D: Rastrigin at 2, 5, 7 dimensions
for nd in (2, 5, 7):
    _register(
        SingleObjectiveProblem(
            name=f"rastrigin_{nd}d",
            func=f.rastrigin,
            bounds=_nd_bounds(nd, -5.12, 5.12),
            known_minimum=0.0,
            family="rastrigin",
            description=f"{nd}D, highly multimodal",
        )
    )

# The rounded 418.9829 constant used by the implementation leaves a small,
# nonzero residual at x = 420.9687463599821.
_SCHWEFEL_MINIMUM_PER_DIMENSION = 1.2727566229386866e-5

# N-D: Schwefel at 2, 5, 7 dimensions
for nd in (2, 5, 7):
    _register(
        SingleObjectiveProblem(
            name=f"schwefel_{nd}d",
            func=f.schwefel,
            bounds=_nd_bounds(nd, -500.0, 500.0),
            known_minimum=nd * _SCHWEFEL_MINIMUM_PER_DIMENSION,
            family="schwefel",
            description=f"{nd}D, deceptive global minimum",
        )
    )
