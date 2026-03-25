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

"""Real-world benchmark: gradient boosted regressor on the diabetes dataset."""

from __future__ import annotations

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Preload dataset once
_DIABETES = load_diabetes()
_X_TRAIN, _X_TEST, _Y_TRAIN, _Y_TEST = train_test_split(
    _DIABETES.data, _DIABETES.target, test_size=0.1, random_state=13
)


def gbr_diabetes(p: dict[str, float]) -> float:
    """Minimize negative R^2 of a GBR on the diabetes dataset.

    Parameters: n_estimators (int), max_depth (int), learning_rate, subsample.
    """
    model = GradientBoostingRegressor(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]),
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        random_state=42,
    )
    model.fit(_X_TRAIN, _Y_TRAIN)
    r2 = model.score(_X_TEST, _Y_TEST)
    return float(-r2)  # Minimize negative R^2 (maximize R^2)
