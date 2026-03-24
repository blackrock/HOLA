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

"""Hypervolume indicator."""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def compute_hv(front: np.ndarray, reference_point: np.ndarray | tuple[float, ...]) -> float:
    """Compute hypervolume dominated by front relative to reference point.

    Higher is better. Returns 0.0 if front is empty.
    """
    if front.size == 0:
        return 0.0
    ref = np.array(reference_point, dtype=float)
    indicator = HV(ref_point=ref)
    return float(indicator(front))
