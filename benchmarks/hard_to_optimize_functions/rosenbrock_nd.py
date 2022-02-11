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
from typing import Iterable, Union, cast

import numpy as np


def rosenbrock(x: Union[float, Iterable[float]]) -> float:
    """https://www.sfu.ca/~ssurjano/rosen.html."""
    if isinstance(x, Iterable):
        x = np.array(x)  # This may consume the iterable (if it's a generator for example)
        if len(x.shape) != 1:
            raise AssertionError("Invalid shape")
    else:
        x = np.array([x])
    x_sq = np.power(x, 2)
    terms = 100 * np.power(x[1:] - x_sq[:-1], 2) + np.power(x[:-1] - 1, 2)
    return cast(float, np.sum(terms))
