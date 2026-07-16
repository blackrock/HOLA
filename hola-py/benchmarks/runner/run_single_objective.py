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

from benchmarks.adapters.base import SingleObjectiveOptimizer
from benchmarks.adapters.hola_adapter import HolaSingleObjectiveAdapter
from benchmarks.adapters.optuna_adapter import OptunaTPEAdapter
from benchmarks.adapters.pymoo_single import (
    ga_adapter,
    hooke_jeeves_adapter,
    pso_adapter,
)
from benchmarks.adapters.random_double import RandomDoubleAdapter
from benchmarks.problems.single_objective import SINGLE_OBJECTIVE_PROBLEMS
from benchmarks.runner.config import RunConfig
from benchmarks.runner.executor import run_single_objective


def get_all_optimizers() -> list[SingleObjectiveOptimizer]:
    """Return the optimizers in deterministic primary-protocol order."""
    return [
        HolaSingleObjectiveAdapter(strategy="random"),
        HolaSingleObjectiveAdapter(strategy="sobol"),
        HolaSingleObjectiveAdapter(strategy="gmm"),
        # Calibration baseline: deliberately spends twice the declared budget.
        RandomDoubleAdapter(),
        OptunaTPEAdapter(),
        ga_adapter(),
        pso_adapter(),
        hooke_jeeves_adapter(),
    ]


def _select_optimizers(
    available: list[SingleObjectiveOptimizer], requested: str | None
) -> list[SingleObjectiveOptimizer]:
    """Select displayed optimizer names in the explicitly requested order."""
    if requested is None:
        return available
    selector = requested.strip()
    if selector == "all":
        return available
    names = [name.strip() for name in selector.split(",")]
    if not names or any(not name for name in names):
        raise ValueError("optimizer names must be a non-empty comma-separated list")

    by_name = {optimizer.name: optimizer for optimizer in available}
    if len(by_name) != len(available):
        raise RuntimeError("primary optimizer display names must be unique")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate optimizer name(s): {', '.join(duplicates)}")

    unknown = [name for name in names if name not in by_name]
    if unknown:
        valid = ", ".join(by_name)
        raise ValueError(
            f"unknown optimizer name(s): {', '.join(unknown)}. Available names: {valid}"
        )
    return [by_name[name] for name in names]


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run single-objective benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--budgets", type=str, default="200,500,1000,2000")
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Comma-separated problem names; default is the synthetic suite, or use 'all'",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default=None,
        help="Comma-separated displayed optimizer names in desired order, or 'all'",
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
    if parsed.problems is None:
        problems = [
            problem
            for problem in SINGLE_OBJECTIVE_PROBLEMS.values()
            if problem.suite == "synthetic"
        ]
    elif parsed.problems != "all":
        names = parsed.problems.split(",")
        problems = [SINGLE_OBJECTIVE_PROBLEMS[n] for n in names]
    else:
        problems = list(SINGLE_OBJECTIVE_PROBLEMS.values())

    try:
        optimizers = _select_optimizers(get_all_optimizers(), getattr(parsed, "optimizers", None))
    except ValueError as error:
        parser.error(str(error))
    run_single_objective(problems, optimizers, config)


if __name__ == "__main__":
    main()
