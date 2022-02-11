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


def michalewicz(x: Union[float, Iterable[float]], m: float = 10) -> float:
    """https://www.sfu.ca/~ssurjano/michal.html."""
    if isinstance(x, Iterable):
        x = np.array(x)  # This may consume the iterable (if it's a generator for example)
        n_dimensions = x.size
        if len(x.shape) != 1:
            raise AssertionError("Invalid shape")
    else:
        n_dimensions = 1
    terms = [np.sin(x) * np.power(np.sin(i * x**2 / np.pi), 2 * m) for i in range(1, n_dimensions + 1)]
    res = -np.sum(terms)
    return cast(float, res)
