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

"""Entry point for single-objective benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.adapters.hola_adapter import HolaSingleObjectiveAdapter
from benchmarks.adapters.igr_adapter import IGRAdapter
from benchmarks.adapters.optuna_adapter import OptunaTPEAdapter
from benchmarks.adapters.pymoo_single import (
    ga_adapter,
    hooke_jeeves_adapter,
    nelder_mead_adapter,
    pso_adapter,
)
from benchmarks.adapters.random_double import RandomDoubleAdapter
from benchmarks.problems.single_objective import SINGLE_OBJECTIVE_PROBLEMS
from benchmarks.runner.config import RunConfig
from benchmarks.runner.executor import run_single_objective


def get_all_optimizers() -> list:
    return [
        HolaSingleObjectiveAdapter(strategy="random"),
        HolaSingleObjectiveAdapter(strategy="sobol"),
        HolaSingleObjectiveAdapter(strategy="gmm"),
        RandomDoubleAdapter(),
        IGRAdapter(spacing=4),
        OptunaTPEAdapter(),
        ga_adapter(),
        pso_adapter(),
        nelder_mead_adapter(),
        hooke_jeeves_adapter(),
    ]


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run single-objective benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--budgets", type=str, default="25,50,75,100,200,500,1000")
    parser.add_argument(
        "--problems", type=str, default=None, help="Comma-separated problem names, or 'all'"
    )
    parser.add_argument("--no-resume", action="store_true")

    parsed = parser.parse_args() if args is None else args

    budgets = [int(b) for b in parsed.budgets.split(",")]
    config = RunConfig(
        output_dir=parsed.output_dir,
        n_runs=parsed.n_runs,
        n_workers=parsed.n_workers,
        budgets=budgets,
        resume=not parsed.no_resume,
    )

    # Select problems
    if parsed.problems and parsed.problems != "all":
        names = parsed.problems.split(",")
        problems = [SINGLE_OBJECTIVE_PROBLEMS[n] for n in names]
    else:
        problems = list(SINGLE_OBJECTIVE_PROBLEMS.values())

    optimizers = get_all_optimizers()
    run_single_objective(problems, optimizers, config)


if __name__ == "__main__":
    main()
