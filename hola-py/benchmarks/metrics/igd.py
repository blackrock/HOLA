# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inverted Generational Distance (IGD)."""

from __future__ import annotations

import numpy as np
from pymoo.indicators.igd import IGD


def compute_igd(front: np.ndarray, true_front: np.ndarray) -> float:
    """Compute IGD from true front to approximation.

    Lower is better. Returns inf if front is empty.
    """
    if front.size == 0:
        return float("inf")
    indicator = IGD(true_front)
    return float(indicator(front))
