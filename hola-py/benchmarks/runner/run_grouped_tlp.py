# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for the explicitly grouped-TLP capability benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.adapters.base import MultiObjectiveOptimizer
from benchmarks.adapters.hola_adapter import HolaGroupedTlpAdapter
from benchmarks.adapters.optuna_adapter import OptunaNSGAIIAdapter
from benchmarks.adapters.pymoo_multi import PymooMOEADAdapter, PymooNSGAIIAdapter
from benchmarks.problems.grouped_tlp import SYNTHETIC_GROUPED_TLP
from benchmarks.problems.registry import GroupedTlpProblem
from benchmarks.runner.config import RunConfig
from benchmarks.runner.executor import run_multi_objective
from benchmarks.runner.run_multi_objective import _select_optimizers


def get_all_optimizers(problem: GroupedTlpProblem) -> list[MultiObjectiveOptimizer]:
    """Return fixed practical configurations for the capability comparison."""
    return [
        HolaGroupedTlpAdapter(problem, strategy="gmm"),
        OptunaNSGAIIAdapter(),
        PymooNSGAIIAdapter(),
        PymooMOEADAdapter(),
    ]


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the grouped-TLP capability benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/grouped_tlp"),
    )
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--budgets", type=str, default="200,500,1000,2000")
    parser.add_argument(
        "--optimizers",
        type=str,
        default=None,
        help="Comma-separated displayed optimizer names in desired order, or 'all'",
    )
    parser.add_argument("--no-resume", action="store_true")
    parsed = parser.parse_args() if args is None else args

    problem = SYNTHETIC_GROUPED_TLP
    try:
        optimizers = _select_optimizers(
            get_all_optimizers(problem),
            getattr(parsed, "optimizers", None),
        )
    except ValueError as error:
        parser.error(str(error))

    config = RunConfig(
        output_dir=parsed.output_dir,
        n_runs=parsed.n_runs,
        n_workers=parsed.n_workers,
        budgets=[int(budget) for budget in parsed.budgets.split(",")],
        resume=not parsed.no_resume,
    )
    run_multi_objective(
        [problem.as_multi_objective_problem()],
        optimizers,
        config,
    )


if __name__ == "__main__":
    main()
