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

"""Dedicated practical HPO campaign with sealed held-out evaluation."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.adapters.base import (
    EvaluationCountError,
    HpoOptimizer,
    assert_exact_evaluations,
    optimizer_configuration,
)
from benchmarks.adapters.hpo import HolaHpoAdapter, OptunaTpeHpoAdapter
from benchmarks.data.manifest import build_campaign_manifest
from benchmarks.data.persistence import ResultStore
from benchmarks.data.seeding import make_hpo_split_seed, make_seed
from benchmarks.problems.hpo import HPO_PROBLEMS, HpoProblem
from benchmarks.runner.config import RunConfig
from benchmarks.runner.executor import (
    _bounded_futures,
    _format_error,
)


def get_all_optimizers() -> list[HpoOptimizer]:
    """Return only native mixed-space optimizers in protocol order."""
    return [
        HolaHpoAdapter("random"),
        HolaHpoAdapter("sobol"),
        HolaHpoAdapter("gmm"),
        OptunaTpeHpoAdapter(),
    ]


def _select_optimizers(
    available: list[HpoOptimizer],
    requested: str | None,
) -> list[HpoOptimizer]:
    if requested is None or requested.strip() == "all":
        return available
    names = [name.strip() for name in requested.split(",")]
    if not names or any(not name for name in names):
        raise ValueError("optimizer names must be a non-empty comma-separated list")
    if len(names) != len(set(names)):
        raise ValueError("duplicate optimizer names are not allowed")
    by_name = {optimizer.name: optimizer for optimizer in available}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(
            f"unknown optimizer name(s): {', '.join(unknown)}. "
            f"Available names: {', '.join(by_name)}"
        )
    return [by_name[name] for name in names]


def _resolved_configuration(
    optimizer: HpoOptimizer,
    budget: int,
) -> tuple[dict[str, Any], Exception | None]:
    try:
        return optimizer_configuration(optimizer, budget), None
    except Exception as error:
        return (
            {
                "adapter": type(optimizer).__name__,
                "name": optimizer.name,
                "configuration_error": _format_error(error),
            },
            error,
        )


def _campaign_configurations(
    optimizers: list[HpoOptimizer],
    budgets: list[int],
) -> list[dict[str, Any]]:
    configurations = []
    for optimizer in optimizers:
        by_budget = []
        for budget in budgets:
            configuration, _ = _resolved_configuration(optimizer, budget)
            by_budget.append({"budget": budget, "configuration": configuration})
        configurations.append({"optimizer": optimizer.name, "by_budget": by_budget})
    return configurations


def _empty_row(
    problem: HpoProblem,
    optimizer: HpoOptimizer,
    budget: int,
    run_id: int,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "problem": problem.name,
        "optimizer": optimizer.name,
        "budget": budget,
        "run_id": run_id,
        "search_seed": make_seed(problem.name, budget, run_id),
        "split_seed": make_hpo_split_seed(problem.name, run_id),
        "status": "error",
        "error": "",
        "optimizer_config": configuration,
        "n_validation_evaluations": None,
        "best_validation_r2": None,
        "best_params": None,
        "validation_trace": None,
        "heldout_test_r2": None,
        "n_heldout_evaluations": 0,
        "train_size": None,
        "validation_size": None,
        "test_size": None,
        "wall_time_seconds": None,
    }


def _run_hpo_one(
    problem: HpoProblem,
    optimizer: HpoOptimizer,
    budget: int,
    run_id: int,
) -> dict[str, Any]:
    """Run validation optimization, then perform exactly one held-out score."""
    configuration, configuration_error = _resolved_configuration(optimizer, budget)
    row = _empty_row(problem, optimizer, budget, run_id, configuration)
    if configuration_error is not None:
        row["error"] = _format_error(configuration_error)
        return row

    evaluator = problem.make_evaluator(int(row["split_seed"]), budget)
    train_size, validation_size, test_size = evaluator.split_sizes
    row.update(
        {
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
        }
    )
    started = time.perf_counter()
    try:
        result = optimizer.optimize(
            problem,
            evaluator.evaluate_validation,
            budget,
            int(row["search_seed"]),
        )
        assert_exact_evaluations(result.n_evaluations, budget, optimizer.name)
        assert_exact_evaluations(evaluator.validation_calls, budget, optimizer.name)
        if len(result.validation_trace) != budget:
            raise EvaluationCountError(len(result.validation_trace), budget, optimizer.name)
        if evaluator.heldout_calls != 0:
            raise RuntimeError("optimizer accessed held-out data during validation search")

        selected_params = problem.normalize_params(result.best_params)
        row.update(
            {
                "n_validation_evaluations": evaluator.validation_calls,
                "best_validation_r2": result.best_validation_value,
                "best_params": selected_params,
                "validation_trace": result.validation_trace,
            }
        )
        heldout_test_r2 = evaluator.evaluate_heldout(selected_params)
        assert_exact_evaluations(evaluator.heldout_calls, 1, f"{optimizer.name} held-out")
        row.update(
            {
                "status": "success",
                "heldout_test_r2": heldout_test_r2,
                "n_heldout_evaluations": evaluator.heldout_calls,
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
    except Exception as error:
        row["error"] = _format_error(error)
        row["n_validation_evaluations"] = evaluator.validation_calls
        row["n_heldout_evaluations"] = evaluator.heldout_calls
        row["wall_time_seconds"] = time.perf_counter() - started
    return row


def _worker_failure(
    problem: HpoProblem,
    optimizer: HpoOptimizer,
    budget: int,
    run_id: int,
    error: Exception,
) -> dict[str, Any]:
    configuration, _ = _resolved_configuration(optimizer, budget)
    row = _empty_row(problem, optimizer, budget, run_id, configuration)
    row["error"] = _format_error(error)
    return row


def _print_progress(done: int, total: int, row: dict[str, Any]) -> None:
    outcome = (
        f"validation R2={row['best_validation_r2']:.4f}, held-out R2={row['heldout_test_r2']:.4f}"
        if row["status"] == "success"
        else f"ERROR: {row['error']}"
    )
    print(
        f"  [{done}/{total}] {row['problem']} / {row['optimizer']} "
        f"/ budget={row['budget']} -> {outcome}"
    )


def run_hpo(
    problems: list[HpoProblem],
    optimizers: list[HpoOptimizer],
    config: RunConfig,
) -> None:
    """Run an immutable, resume-safe practical HPO campaign."""
    manifest = build_campaign_manifest(
        run_kind="hpo",
        budgets=config.budgets,
        n_runs=config.n_runs,
        problem_names=[problem.name for problem in problems],
        optimizer_names=[optimizer.name for optimizer in optimizers],
        optimizer_configurations=_campaign_configurations(optimizers, config.budgets),
        campaign_configuration={
            "problems": [problem.configuration() for problem in problems],
            "selection": "maximum fixed-validation R2",
            "final_evaluation": "single train+validation refit and untouched test score",
            "seed_derivation": {
                "search_seed": "first 32 bits of SHA-256(problem:budget:run_id)",
                "split_seed": "first 32 bits of SHA-256(hpo-split:problem:run_id)",
                "pairing": (
                    "search seed paired across optimizers within problem+budget+run_id; "
                    "split seed paired across optimizers and fixed across budgets within "
                    "problem+run_id"
                ),
            },
        },
    )
    store = ResultStore(config.output_dir)
    store.prepare_campaign(manifest, resume=config.resume)
    completed = store.completed_hpo_runs() if config.resume else set()

    work: list[tuple[HpoProblem, HpoOptimizer, int, int]] = []
    for problem in problems:
        for optimizer in optimizers:
            for budget in config.budgets:
                for run_id in range(config.n_runs):
                    key = (problem.name, optimizer.name, budget, run_id)
                    if key not in completed:
                        work.append((problem, optimizer, budget, run_id))

    total = len(work)
    if total == 0:
        print("All practical HPO runs already completed.")
        return
    print(
        f"Running {total} practical HPO evaluations "
        f"({len(problems)} problems x {len(optimizers)} optimizers)"
    )

    done = 0
    if config.effective_workers == 1:
        for problem, optimizer, budget, run_id in work:
            row = _run_hpo_one(problem, optimizer, budget, run_id)
            store.append_hpo(row)
            done += 1
            _print_progress(done, total, row)
    else:
        with ProcessPoolExecutor(max_workers=config.effective_workers) as executor:
            for future, item in _bounded_futures(
                executor,
                _run_hpo_one,
                work,
                max_in_flight=2 * config.effective_workers,
            ):
                problem, optimizer, budget, run_id = item
                try:
                    row = future.result()
                except Exception as error:
                    row = _worker_failure(problem, optimizer, budget, run_id, error)
                store.append_hpo(row)
                done += 1
                _print_progress(done, total, row)


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the practical mixed-space HPO benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/hpo"))
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Paired replications (default 1 for the protocol sentinel campaign)",
    )
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--budgets", type=str, default="25,50,100")
    parser.add_argument(
        "--optimizers",
        type=str,
        default=None,
        help="Comma-separated displayed optimizer names in desired order, or 'all'",
    )
    parser.add_argument("--no-resume", action="store_true")
    parsed = parser.parse_args() if args is None else args

    budgets = [int(value) for value in parsed.budgets.split(",")]
    if any(budget not in {25, 50, 100} for budget in budgets):
        parser.error("the preregistered practical HPO budgets are 25, 50, and 100")
    try:
        optimizers = _select_optimizers(get_all_optimizers(), parsed.optimizers)
    except ValueError as error:
        parser.error(str(error))
    config = RunConfig(
        output_dir=parsed.output_dir,
        n_runs=parsed.n_runs,
        n_workers=parsed.n_workers,
        budgets=budgets,
        resume=not parsed.no_resume,
    )
    run_hpo(list(HPO_PROBLEMS.values()), optimizers, config)


if __name__ == "__main__":
    main()
