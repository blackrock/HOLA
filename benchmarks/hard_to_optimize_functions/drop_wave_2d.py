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


def drop_wave(x1: float, x2: float) -> float:
    """https://www.sfu.ca/~ssurjano/drop.html."""
    sq_sum = x1**2 + x2**2
    numerator = 1 + np.cos(12 * np.sqrt(sq_sum))
    denominator = 0.5 * sq_sum + 2
    return cast(float, -numerator / denominator)


def drop_wave_np(x: npt.NDArray[np.floating]) -> float:
    if len(x) != 2:
        raise AssertionError("Exactly 2 items expected")
    return drop_wave(x[0], x[1])


def drop_wave_wrapper(x: tuple[float, ...]) -> float:
    return drop_wave(*x)
