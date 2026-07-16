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

"""Fixed, problem-defined normalization for multi-objective metrics."""

from __future__ import annotations

import numpy as np


def normalize_objectives(
    points: np.ndarray,
    ideal_point: np.ndarray | tuple[float, ...],
    reference_point: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Normalize objective coordinates from the ideal-reference box.

    The transformation is fixed by the problem definition: the ideal point
    maps to zero and the reporting reference point maps to one. Coordinates
    are not clipped, so points outside that box remain visible to the metrics.
    """
    ideal = np.asarray(ideal_point, dtype=float)
    reference = np.asarray(reference_point, dtype=float)
    if ideal.ndim != 1 or reference.ndim != 1 or ideal.shape != reference.shape:
        raise ValueError("ideal and reference points must be same-length vectors")
    if not np.all(np.isfinite(ideal)) or not np.all(np.isfinite(reference)):
        raise ValueError("ideal and reference points must be finite")

    scale = reference - ideal
    if np.any(scale <= 0.0):
        raise ValueError("each reference coordinate must be greater than its ideal")

    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return np.empty((0, len(ideal)), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(ideal):
        raise ValueError(f"objective points must have shape (n, {len(ideal)})")
    return (values - ideal) / scale
