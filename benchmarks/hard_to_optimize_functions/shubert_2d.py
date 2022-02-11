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


def shubert(x1: float, x2: float) -> float:
    """https://www.sfu.ca/~ssurjano/shubert.html."""
    factor_1 = np.sum([i * np.cos((i + 1) * x1 + i) for i in range(1, 6)])
    factor_2 = np.sum([i * np.cos((i + 1) * x2 + i) for i in range(1, 6)])
    return cast(float, factor_1 * factor_2)


def shubert_np(x: npt.NDArray[np.floating]) -> float:
    if len(x) != 2:
        raise AssertionError("Exactly 2 items expected")
    return shubert(x[0], x[1])
