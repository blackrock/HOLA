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

"""Basic optimization walkthrough.

Demonstrates the core HOLA workflow with TLP scoring:
  1. Minimizing 1D Forrester function using study.run()
  2. Minimizing 2D Branin function using ask/tell loop with Sobol

TLP (Target-Limit-Priority) scoring normalizes each objective to [0, 1]:
  - At or below target: score = 0 (satisfied)
  - Between target and limit: score in (0, 1) (acceptable)
  - Beyond limit: score = inf (infeasible)
"""

from benchmarks.functions.single_objective import branin, forrester
from hola import Minimize, Real, Space, Study


def example_1d():
    """Optimize the 1D Forrester function."""
    print("=" * 60)
    print("Example 1: Forrester function (1D)")
    print("  Known minimum: -6.0267 at x ≈ 0.7572")
    print("=" * 60)

    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("value", target=-6.0, limit=0.0)],
        strategy="sobol",
    )

    study.run(lambda p: {"value": forrester(p)}, n_trials=50)

    best = study.top_k(1)[0]
    print(f"\n  Best found: {best.metrics['value']:.4f}")
    print(f"  TLP score:  {best.scores['value']:.4f}  (0 = at target, 1 = at limit)")
    print(f"  At x = {best.params['x']:.4f}")
    print(f"  Gap from known optimum: {best.metrics['value'] - (-6.0267):.4f}")
    print()


def example_2d():
    """Optimize the 2D Branin function with ask/tell."""
    print("=" * 60)
    print("Example 2: Branin function (2D) — ask/tell loop")
    print("  Known minimum: 0.397887")
    print("=" * 60)

    study = Study(
        space=Space(
            x1=Real(-5.0, 10.0),
            x2=Real(0.0, 15.0),
        ),
        objectives=[Minimize("value", target=0.5, limit=50.0)],
        strategy="sobol",
    )

    # Manual ask/tell loop
    for i in range(100):
        trial = study.ask()
        result = branin(trial.params)
        study.tell(trial.trial_id, {"value": result})

        if (i + 1) % 25 == 0:
            best = study.top_k(1)[0]
            print(
                f"  Trial {i + 1:3d}: best = {best.metrics['value']:.6f}"
                f"  (TLP: {best.scores['value']:.4f})"
            )

    best = study.top_k(1)[0]
    print(f"\n  Final best: {best.metrics['value']:.6f}  (TLP: {best.scores['value']:.4f})")
    print(f"  At x1={best.params['x1']:.4f}, x2={best.params['x2']:.4f}")
    print(f"  Gap: {best.metrics['value'] - 0.397887:.6f}")
    print()


if __name__ == "__main__":
    example_1d()
    example_2d()
