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

from benchmarks.metrics.normalization import normalize_objectives


def compute_hv(front: np.ndarray, reference_point: np.ndarray | tuple[float, ...]) -> float:
    """Compute hypervolume dominated by front relative to reference point.

    Higher is better. Returns 0.0 if front is empty.
    """
    values = np.asarray(front, dtype=float)
    if values.size == 0:
        return 0.0
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.size == 0:
        return 0.0
    ref = np.array(reference_point, dtype=float)
    indicator = HV(ref_point=ref)
    return float(indicator(values))


def compute_normalized_hv_gap(
    front: np.ndarray,
    true_front: np.ndarray | None,
    ideal_point: np.ndarray | tuple[float, ...],
    reference_point: np.ndarray | tuple[float, ...],
    *,
    reference_hypervolume: float | None = None,
) -> float:
    """Return hypervolume shortfall as a fraction of the ideal-reference box.

    The approximation is transformed with fixed problem-defined coordinates.
    The true-front hypervolume may be supplied from an analytic or cached
    problem reference; otherwise it is computed from ``true_front``. The
    normalized reference point is the all-ones vector and the unit box has
    volume one. A zero gap is best. Small negative differences caused by a
    finite reference-front approximation are truncated to zero.
    """
    normalized_front = normalize_objectives(front, ideal_point, reference_point)
    normalized_reference = np.ones(len(np.asarray(reference_point)), dtype=float)
    if reference_hypervolume is None:
        if true_front is None:
            raise ValueError("a true front or known reference hypervolume is required")
        normalized_true_front = normalize_objectives(true_front, ideal_point, reference_point)
        if normalized_true_front.size == 0:
            raise ValueError("a non-empty true front is required for hypervolume gap")
        reference_hypervolume = compute_hv(normalized_true_front, normalized_reference)
    if not np.isfinite(reference_hypervolume) or not 0.0 < reference_hypervolume <= 1.0:
        raise ValueError("normalized reference hypervolume must be in (0, 1]")
    approximation_hv = compute_hv(normalized_front, normalized_reference)
    return max(0.0, reference_hypervolume - approximation_hv)
