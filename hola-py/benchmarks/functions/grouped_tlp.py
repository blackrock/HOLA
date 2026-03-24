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

"""Grouped TLP benchmark functions.

These test the full priority-group pipeline where each group contains
multiple objectives with TLP scalarization.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split


def _extract_vec(p: dict[str, float]) -> np.ndarray:
    keys = sorted(k for k in p if k.startswith("x"))
    return np.array([p[k] for k in keys])


# ---------------------------------------------------------------------------
# Synthetic grouped problem: 4 objectives in 2 groups
# ---------------------------------------------------------------------------


def synthetic_grouped(p: dict[str, float]) -> dict[str, float]:
    """Synthetic grouped TLP benchmark.

    4 raw objectives partitioned into 2 groups of 2.
    Decision variables: x0..x4 in [0, 1].

    Group 1: f1 = ||x||^2, f2 = ||x - e1||^2
    Group 2: f3 = ||x - e2||^2, f4 = ||x - e1 - e2||^2

    where e1, e2 are the first two standard basis vectors.
    """
    x = _extract_vec(p)
    n = len(x)

    e1 = np.zeros(n)
    e1[0] = 1.0
    e2 = np.zeros(n)
    e2[1] = 1.0

    f1 = float(np.sum(x**2))
    f2 = float(np.sum((x - e1) ** 2))
    f3 = float(np.sum((x - e2) ** 2))
    f4 = float(np.sum((x - e1 - e2) ** 2))

    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4}


# ---------------------------------------------------------------------------
# ML-inspired grouped problem: performance vs resource
# ---------------------------------------------------------------------------

# Preload dataset once at module level
_DIABETES = load_diabetes()
_X_TRAIN, _X_TEST, _Y_TRAIN, _Y_TEST = train_test_split(
    _DIABETES.data, _DIABETES.target, test_size=0.2, random_state=42
)


def ml_grouped(p: dict[str, float]) -> dict[str, float]:
    """ML-inspired grouped TLP benchmark.

    Hyper-parameters: n_estimators, max_depth, learning_rate, subsample.

    Performance group: {accuracy (R^2), pseudo-F1}
    Resource group: {training_time, model_size (n_estimators * max_depth)}
    """
    n_estimators = int(p["n_estimators"])
    max_depth = int(p["max_depth"])
    learning_rate = p["learning_rate"]
    subsample = p["subsample"]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
    )

    t0 = time.perf_counter()
    model.fit(_X_TRAIN, _Y_TRAIN)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(_X_TEST)
    r2 = r2_score(_Y_TEST, y_pred)

    # Pseudo-F1: threshold regression predictions into binary (above/below median)
    median_y = float(np.median(_Y_TEST))
    y_true_bin = (_Y_TEST > median_y).astype(int)  # noqa: SIM300
    y_pred_bin = (y_pred > median_y).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0.0)

    model_size = float(n_estimators * max_depth)

    return {
        "r2": r2,
        "f1": f1,
        "train_time": train_time,
        "model_size": model_size,
    }
