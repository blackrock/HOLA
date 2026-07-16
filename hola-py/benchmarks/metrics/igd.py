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

from benchmarks.metrics.normalization import normalize_objectives


def compute_igd(front: np.ndarray, true_front: np.ndarray) -> float:
    """Compute IGD from true front to approximation.

    Lower is better. Returns inf if front is empty.
    """
    values = np.asarray(front, dtype=float)
    reference = np.asarray(true_front, dtype=float)
    if values.size == 0:
        return float("inf")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)
    values = values[np.all(np.isfinite(values), axis=1)]
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    if values.size == 0:
        return float("inf")
    if reference.size == 0:
        raise ValueError("a non-empty finite true front is required for IGD")
    indicator = IGD(reference)
    return float(indicator(values))


def compute_normalized_igd(
    front: np.ndarray,
    true_front: np.ndarray,
    ideal_point: np.ndarray | tuple[float, ...],
    reference_point: np.ndarray | tuple[float, ...],
) -> float:
    """Compute IGD after fixed ideal-reference objective normalization."""
    normalized_front = normalize_objectives(front, ideal_point, reference_point)
    normalized_true_front = normalize_objectives(true_front, ideal_point, reference_point)
    return compute_igd(normalized_front, normalized_true_front)
