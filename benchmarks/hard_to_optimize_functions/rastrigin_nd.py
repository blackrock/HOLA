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

from typing import Iterable, Union, cast, overload

import numpy as np
import numpy.typing as npt


@overload
def _rastrigin_term(x: float) -> float:
    ...


@overload
def _rastrigin_term(x: Iterable[float]) -> npt.NDArray[np.floating]:
    ...


def _rastrigin_term(x: Union[float, Iterable[float]]) -> Union[float, npt.NDArray[np.floating]]:
    x_array: Union[float, npt.NDArray[np.floating]] = np.array(x) if isinstance(x, Iterable) else x
    return x_array**2 - 10 * np.cos(2 * np.pi * x_array)


def rastrigin(x: Union[float, Iterable[float]]) -> float:
    """https://www.sfu.ca/~ssurjano/rastr.html."""
    if isinstance(x, Iterable):
        x = np.array(x)  # This may consume the iterable (if it's a generator for example)
        n_dimensions = x.size
        if len(x.shape) != 1:
            raise AssertionError("Invalid shape")
    else:
        n_dimensions = 1
    return cast(float, 10 * n_dimensions + np.sum(_rastrigin_term(x)))
