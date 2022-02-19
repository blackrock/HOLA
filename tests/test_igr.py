# Copyright 2021 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence

from pytest import approx

from hola.igr_algorithm import Evaluation, Hypercube, IterativeGridRefinement, Side
from hola.params import ParamConfig


def test_hypercube_dimensions() -> None:
    unit_side = Side(0, 1)
    assert unit_side.length == approx(1)
    assert unit_side.get_lattices(4) == approx([0, 0.25, 0.5, 0.75, 1])
    assert unit_side.get_lattices(3) == approx([0.0, 0.3333333333333333, 0.6666666666666666, 1.0])
    assert unit_side.shrink_around(0.5, 4) == Side(0.375, 0.625)
    assert unit_side.shrink_around(0, 4) == Side(lower=0, upper=0.125)  # extreme
    assert unit_side.shrink_around(0.75, 4) == Side(lower=0.625, upper=0.875)
    assert unit_side.shrink_around(0.9, 4) == Side(lower=0.775, upper=1)  # extreme
    other_side = Side(0.2, 0.8)
    assert other_side.length == approx(0.6)
    assert other_side.get_lattices(3) == approx([0.2, 0.4, 0.6, 0.8])
    assert other_side.get_lattices(4) == approx([0.2, 0.35, 0.5, 0.65, 0.8])
    assert unit_side.shrink_around(0.5, 4) == Side(lower=0.375, upper=0.625)
    assert unit_side.shrink_around(0.35, 4) == Side(lower=0.22499999999999998, upper=0.475)
    assert unit_side.shrink_around(0.2, 4) == Side(lower=0.07500000000000001, upper=0.325)
    assert unit_side.shrink_around(0.4, 5) == Side(lower=0.30000000000000004, upper=0.5)
    assert unit_side.shrink_around(0.6, 5) == Side(lower=0.5, upper=0.7)


def test_hypercube() -> None:
    cube = Hypercube({"a": Side(0, 1), "b": Side(0.3, 0.4)})
    assert cube.get_lattices(4) == [
        {"a": 0.0, "b": 0.3},
        {"a": 0.0, "b": 0.325},
        {"a": 0.0, "b": 0.35},
        {"a": 0.0, "b": 0.375},
        {"a": 0.0, "b": 0.4},
        {"a": 0.25, "b": 0.3},
        {"a": 0.25, "b": 0.325},
        {"a": 0.25, "b": 0.35},
        {"a": 0.25, "b": 0.375},
        {"a": 0.25, "b": 0.4},
        {"a": 0.5, "b": 0.3},
        {"a": 0.5, "b": 0.325},
        {"a": 0.5, "b": 0.35},
        {"a": 0.5, "b": 0.375},
        {"a": 0.5, "b": 0.4},
        {"a": 0.75, "b": 0.3},
        {"a": 0.75, "b": 0.325},
        {"a": 0.75, "b": 0.35},
        {"a": 0.75, "b": 0.375},
        {"a": 0.75, "b": 0.4},
        {"a": 1.0, "b": 0.3},
        {"a": 1.0, "b": 0.325},
        {"a": 1.0, "b": 0.35},
        {"a": 1.0, "b": 0.375},
        {"a": 1.0, "b": 0.4},
    ]
    assert cube.shrink_around({"a": 0.75, "b": 0.65}, spacing=4) == Hypercube(
        sides={"a": Side(lower=0.625, upper=0.875), "b": Side(lower=0.6375000000000001, upper=0.6625)},
    )


def test_igr_tune() -> None:
    def quadratic(x: Sequence[float]) -> float:
        return x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2

    params = {f"x{i}": ParamConfig(min=-10, max=10) for i in range(1, 3)}
    igr1 = IterativeGridRefinement(params, spacing=2)
    assert igr1.tune(quadratic, max_iterations=9) == Evaluation(params={"x1": -10.0, "x2": 10.0}, val=0.0)
    igr2 = IterativeGridRefinement(params, spacing=4)
    assert igr2.tune(quadratic, max_iterations=75) == Evaluation(params={"x1": -9.6875, "x2": 9.6875}, val=0.0)
    igr3 = IterativeGridRefinement(params, spacing=10)
    assert igr3.tune(quadratic, max_iterations=363) == Evaluation(
        params={"x1": -9.91, "x2": 9.909999999999997}, val=-1.4210854715202004e-14
    )
