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

"""Spacing metric for Pareto front uniformity."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def compute_spacing(front: np.ndarray) -> float:
    """Compute spacing metric (std of nearest-neighbor distances).

    Lower is better (more uniform). Returns 0.0 for fewer than 2 points.
    """
    if len(front) < 2:
        return 0.0
    dists = cdist(front, front)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    return float(np.std(nn_dists))
