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

"""Run-scoped objective implementation for the practical diabetes HPO task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from benchmarks.problems.hpo import HpoProblem


class DiabetesGbrEvaluator:
    """Deterministic validation objective with a sealed held-out test split."""

    def __init__(
        self,
        problem: HpoProblem,
        split_seed: int,
        validation_budget: int,
    ) -> None:
        self.problem = problem
        self.split_seed = split_seed
        self.validation_budget = validation_budget
        self.validation_calls = 0
        self.heldout_calls = 0

        dataset = load_diabetes()
        split_state, validation_state, model_state = np.random.SeedSequence(
            split_seed
        ).generate_state(3)
        train_validation_x, self._test_x, train_validation_y, self._test_y = train_test_split(
            dataset.data,
            dataset.target,
            test_size=0.20,
            random_state=int(split_state),
        )
        self._train_x, self._validation_x, self._train_y, self._validation_y = train_test_split(
            train_validation_x,
            train_validation_y,
            test_size=0.25,
            random_state=int(validation_state),
        )
        self._model_seed = int(model_state)

    @property
    def split_sizes(self) -> tuple[int, int, int]:
        return len(self._train_y), len(self._validation_y), len(self._test_y)

    def _model(self, params: dict[str, Any]) -> GradientBoostingRegressor:
        normalized = self.problem.normalize_params(params)
        return GradientBoostingRegressor(
            n_estimators=normalized["n_estimators"],
            max_depth=normalized["max_depth"],
            learning_rate=normalized["learning_rate"],
            subsample=normalized["subsample"],
            loss=normalized["loss"],
            random_state=self._model_seed,
        )

    def evaluate_validation(self, params: dict[str, Any]) -> float:
        """Fit on training data and score only the fixed validation set."""
        if self.validation_calls >= self.validation_budget:
            raise RuntimeError("validation evaluation budget is exhausted")
        self.validation_calls += 1
        model = self._model(params)
        model.fit(self._train_x, self._train_y)
        return float(model.score(self._validation_x, self._validation_y))

    def evaluate_heldout(self, params: dict[str, Any]) -> float:
        """Refit the selected configuration once, then score the sealed test set."""
        if self.validation_calls != self.validation_budget:
            raise RuntimeError(
                "held-out evaluation is available only after the validation budget is complete"
            )
        if self.heldout_calls != 0:
            raise RuntimeError("held-out test data may be evaluated only once")
        self.heldout_calls += 1
        model = self._model(params)
        train_validation_x = np.concatenate([self._train_x, self._validation_x], axis=0)
        train_validation_y = np.concatenate([self._train_y, self._validation_y], axis=0)
        model.fit(train_validation_x, train_validation_y)
        return float(model.score(self._test_x, self._test_y))
