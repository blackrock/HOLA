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
from typing import cast

import numpy as np
import numpy.typing as npt


def branin(
    x1: float,
    x2: float,
    a: float = 1,
    b: float = 5.1 / (4 * np.pi**2),
    c: float = 5 / np.pi,
    r: float = 6,
    s: float = 10,
    t: float = 1 / (8 * np.pi),
) -> float:
    """https://www.sfu.ca/~ssurjano/branin.html."""
    res = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return cast(float, res)


def branin_np(x: npt.NDArray[np.floating]) -> float:
    """https://www.sfu.ca/~ssurjano/branin.html."""
    if len(x) != 2:
        raise AssertionError("Exactly 2 items expected")
    return branin(x[0], x[1])
