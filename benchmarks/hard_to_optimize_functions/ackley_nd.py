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

from typing import Iterable, Union, cast

import numpy as np


def ackley(x: Union[float, Iterable[float]], a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> float:
    """https://www.sfu.ca/~ssurjano/ackley.html."""
    if isinstance(x, Iterable):
        x = np.array(x)  # This may consume the iterable (if it's a generator for example)
        n_dimensions = x.size
        if len(x.shape) != 1:
            raise AssertionError("Invalid shape")
    else:
        n_dimensions = 1
    term_1 = a * np.exp(-b * np.sqrt((1 / n_dimensions) * np.sum(x * x)))
    term_2 = np.exp((1 / n_dimensions) * np.sum(np.cos(c * x)))
    res = -term_1 - term_2 + a + np.exp(1)
    return cast(float, res)
