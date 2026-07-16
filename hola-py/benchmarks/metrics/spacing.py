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

from benchmarks.metrics.normalization import normalize_objectives


def compute_spacing(front: np.ndarray) -> float:
    """Compute spacing metric (std of nearest-neighbor distances).

    Lower is better (more uniform). Spacing is a secondary diagnostic and is
    undefined for fewer than three finite, distinct points.
    """
    values = np.asarray(front, dtype=float)
    if values.size == 0:
        return float("nan")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values = values[np.all(np.isfinite(values), axis=1)]
    values = np.unique(values, axis=0)
    if len(values) < 3:
        return float("nan")
    dists = cdist(values, values)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    return float(np.std(nn_dists))


def compute_normalized_spacing(
    front: np.ndarray,
    ideal_point: np.ndarray | tuple[float, ...],
    reference_point: np.ndarray | tuple[float, ...],
) -> float:
    """Compute spacing after fixed ideal-reference objective normalization."""
    return compute_spacing(normalize_objectives(front, ideal_point, reference_point))
