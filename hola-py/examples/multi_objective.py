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

"""Multi-objective optimization with TLP and priority groups.

Demonstrates optimizing two conflicting objectives (error vs latency)
using TLP scoring. Each objective is assigned to a separate group via
the `group` parameter, so HOLA computes a Pareto front over the two
TLP-normalized group costs rather than raw values.

The `priority` parameter is the per-objective weight (slope P_i) in
the TLP formula; `group` controls which objectives are combined into
the same Pareto axis.
"""

from hola_opt import Integer, Minimize, Real, Space, Study


def model_objective(params: dict) -> dict:
    """Simulates a model with accuracy/latency trade-off.

    More layers → better accuracy but higher latency.
    Larger hidden size → better accuracy but higher latency.
    Dropout helps accuracy up to a point, hurts with too much.
    """
    layers = params["layers"]
    hidden = params["hidden_size"]
    dropout = params["dropout"]
    lr = params["lr"]

    # Simulate accuracy (lower is better → "error rate")
    base_error = 0.5
    layer_benefit = -0.04 * min(layers, 8) + 0.02 * max(layers - 8, 0)
    hidden_benefit = -0.001 * hidden
    dropout_effect = -0.1 * dropout + 0.3 * dropout**2
    lr_effect = 2.0 * (lr - 0.001) ** 2
    error = max(0.01, base_error + layer_benefit + hidden_benefit + dropout_effect + lr_effect)

    # Simulate latency (ms)
    latency = 10 + 5 * layers + 0.02 * hidden + 0.5 * layers * hidden / 100

    return {"error": error, "latency": latency}


def main():
    print("=" * 60)
    print("Multi-Objective Optimization: Error vs Latency")
    print("=" * 60)

    study = Study(
        space=Space(
            layers=Integer(1, 12),
            hidden_size=Real(32.0, 512.0),
            dropout=Real(0.0, 0.5),
            lr=Real(0.0001, 0.01),
        ),
        objectives=[
            Minimize("error", target=0.05, limit=0.5, group="quality"),
            Minimize("latency", target=20.0, limit=100.0, group="cost"),
        ],
        strategy="sobol",
    )

    study.run(model_objective, n_trials=200)

    # Show Pareto front
    front = study.pareto_front()
    print(f"\n  Pareto front: {len(front)} trials")
    print(
        f"  {'Error':>8s}  {'Latency':>10s}  {'TLP error':>10s}"
        f"  {'TLP latency':>12s}  {'Layers':>6s}  {'Hidden':>6s}"
    )
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 12}  {'─' * 6}  {'─' * 6}")
    for t in front[:8]:
        print(
            f"  {t.metrics['error']:8.4f}  {t.metrics['latency']:10.1f}  "
            f"{t.scores['error']:10.4f}  {t.scores['latency']:12.4f}  "
            f"{t.params['layers']:>6}  {t.params['hidden_size']:6.0f}"
        )
    if len(front) > 8:
        print(f"  ... and {len(front) - 8} more")

    print(f"\n  Completed {study.trial_count()} trials")


if __name__ == "__main__":
    main()
