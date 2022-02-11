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
from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt


def egg_holder(x1: float, x2: float) -> float:
    """https://www.sfu.ca/~ssurjano/egg.html."""
    term_1 = (x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
    term_2 = x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return cast(float, -term_1 - term_2)


def egg_holder_np(x: npt.NDArray[np.floating]) -> float:
    if len(x) != 2:
        raise AssertionError("Exactly 2 items expected")
    return egg_holder(x[0], x[1])


def egg_holder_wrapper(x: tuple[float, ...]) -> float:
    return egg_holder(*x)
