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

"""ML hyperparameter tuning example.

Tunes a sklearn GradientBoostingRegressor on a synthetic dataset
using HOLA. Demonstrates Real, Integer, and Categorical parameter
types in a realistic ML workflow.

Ported from HOLA's analysis/optimizers/benchmarks/boosted_regressor.py.
"""

from hola import Categorical, Integer, Minimize, Real, Space, Study

try:
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def train_and_evaluate(params: dict) -> dict:
    """Train a GBM with the given hyperparameters and return CV loss."""
    X, y = make_friedman1(n_samples=500, noise=1.0, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["lr"],
        subsample=params["subsample"],
        loss=params["loss"],
        random_state=42,
    )

    # 3-fold CV, negative MSE (higher is better in sklearn)
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")
    mse = -scores.mean()
    return {"mse": mse}


def main():
    if not _HAS_SKLEARN:
        print("sklearn not installed. Install with: pip install scikit-learn")
        print("Skipping ML hyperparameter example.")
        return

    print("=" * 60)
    print("ML Hyperparameter Tuning: GradientBoostingRegressor")
    print("=" * 60)

    study = Study(
        space=Space(
            lr=Real(0.001, 0.3, scale="log10"),
            n_estimators=Integer(50, 500),
            max_depth=Integer(2, 8),
            subsample=Real(0.5, 1.0),
            loss=Categorical(["squared_error", "absolute_error", "huber"]),
        ),
        objectives=[Minimize("mse", target=5.0, limit=50.0)],
        strategy="sobol",
    )

    n_trials = 30
    print(f"\n  Running {n_trials} trials (this may take a minute)...")
    for i in range(n_trials):
        trial = study.ask()
        result = train_and_evaluate(trial.params)
        study.tell(trial.trial_id, result)

        if (i + 1) % 10 == 0:
            best = study.top_k(1)[0]
            print(
                f"  Trial {i + 1:3d}: best MSE = {best.metrics['mse']:.4f}"
                f"  (TLP: {best.scores['mse']:.4f})"
            )

    best = study.top_k(1)[0]
    print(f"\n  Best MSE:        {best.metrics['mse']:.4f}  (TLP: {best.scores['mse']:.4f})")
    print(f"  Learning rate:   {best.params['lr']:.6f}")
    print(f"  N estimators:    {best.params['n_estimators']}")
    print(f"  Max depth:       {best.params['max_depth']}")
    print(f"  Subsample:       {best.params['subsample']:.4f}")
    print(f"  Loss function:   {best.params['loss']}")


if __name__ == "__main__":
    main()
