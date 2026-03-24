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

"""Categorical parameter demo.

Shows how to use Categorical, Real, and Integer parameters
together in a single study, simulating an optimizer hyperparameter
search problem.
"""

from hola import Categorical, Integer, Minimize, Real, Space, Study


def simulated_training(params: dict) -> dict:
    """Fake training function that returns a loss depending on the optimizer choice."""
    optimizer = params["optimizer"]
    lr = params["lr"]
    layers = params["layers"]

    # Simulate different optimizer behaviors
    base_loss = {
        "adam": 0.3,
        "sgd": 0.5,
        "rmsprop": 0.35,
        "adamw": 0.28,
    }[optimizer]

    # Optimal lr depends on optimizer
    optimal_lr = {"adam": 1e-3, "sgd": 1e-2, "rmsprop": 5e-4, "adamw": 1e-3}[optimizer]
    lr_penalty = 2.0 * abs(lr - optimal_lr) / optimal_lr

    # More layers help, but diminishing returns + overfitting
    layer_effect = -0.02 * min(layers, 6) + 0.01 * max(layers - 6, 0)

    loss = base_loss + lr_penalty + layer_effect
    return {"loss": loss}


def main():
    print("=" * 60)
    print("Categorical + Mixed Space Demo")
    print("  Find the best (optimizer, lr, layers) combo")
    print("=" * 60)

    study = Study(
        space=Space(
            optimizer=Categorical(["adam", "sgd", "rmsprop", "adamw"]),
            lr=Real(1e-4, 0.1, scale="log10"),
            layers=Integer(1, 10),
        ),
        objectives=[Minimize("loss", target=0.3, limit=2.0)],
        strategy="sobol",
    )

    study.run(simulated_training, n_trials=100)

    best = study.top_k(1)[0]
    print(f"\n  Best loss:     {best.metrics['loss']:.4f}  (TLP: {best.scores['loss']:.4f})")
    print(f"  Optimizer:     {best.params['optimizer']}")
    print(f"  Learning rate: {best.params['lr']:.6f}")
    print(f"  Layers:        {best.params['layers']}")
    print()

    print("  Top 5 trials:")
    print(f"  {'Loss':>10s}  {'TLP':>6s}  {'Optimizer':>10s}  {'LR':>10s}  {'Layers':>6s}")
    print(f"  {'—' * 10}  {'—' * 6}  {'—' * 10}  {'—' * 10}  {'—' * 6}")
    for t in study.top_k(5):
        print(
            f"  {t.metrics['loss']:10.4f}  {t.scores['loss']:6.3f}  {t.params['optimizer']:>10s}  "
            f"{t.params['lr']:10.6f}  {t.params['layers']:>6}"
        )


if __name__ == "__main__":
    main()
