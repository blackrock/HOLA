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


def forrester(x1: float) -> float:
    """https://www.sfu.ca/~ssurjano/forretal08.html."""
    return cast(float, np.power(6 * x1 - 2, 2) * np.sin(12 * x1 - 4))


def forrester_np(x: npt.NDArray[np.floating]) -> float:
    if len(x) != 1:
        raise AssertionError("Exactly 1 items expected")
    return forrester(x[0])
