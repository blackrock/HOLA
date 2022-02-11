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


def six_hump_camel(x1: float, x2: float) -> float:
    """https://www.sfu.ca/~ssurjano/camel6.html."""
    term1 = (4 - 2.1 * np.power(x1, 2) + np.power(x1, 4) / 3) * np.power(x1, 2)
    term2 = x1 * x2
    term3 = (-4 + 4 * np.power(x2, 2)) * np.power(x2, 2)
    return cast(float, term1 + term2 + term3)


def six_hump_camel_np(x: npt.NDArray[np.floating]) -> float:
    if len(x) != 2:
        raise AssertionError("Exactly 2 items expected")
    return six_hump_camel(x[0], x[1])
