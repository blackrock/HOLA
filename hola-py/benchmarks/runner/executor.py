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

"""Parallel executor for benchmark runs."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from typing import Any, TypeAlias

from benchmarks.adapters.base import (
    EvaluationCountError,
    MultiObjectiveOptimizer,
    SingleObjectiveOptimizer,
    optimizer_configuration,
)
from benchmarks.data.manifest import build_campaign_manifest
from benchmarks.data.persistence import ResultStore
from benchmarks.data.seeding import make_seed
from benchmarks.metrics.hypervolume import compute_normalized_hv_gap
from benchmarks.metrics.igd import compute_normalized_igd
from benchmarks.metrics.spacing import compute_normalized_spacing
from benchmarks.problems.registry import MultiObjectiveProblem, SingleObjectiveProblem
from benchmarks.runner.config import RunConfig

WorkItem: TypeAlias = tuple[Any, ...]


def _format_error(error: Exception) -> str:
    """Return a compact, stable error description for a result row."""
    return f"{type(error).__name__}: {error}"


def _resolved_configuration(
    optimizer: SingleObjectiveOptimizer | MultiObjectiveOptimizer,
    budget: int,
) -> tuple[dict[str, Any], Exception | None]:
    """Resolve an adapter configuration without losing a run to configuration failure."""
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
    optimizers: list[SingleObjectiveOptimizer] | list[MultiObjectiveOptimizer],
    budgets: list[int],
) -> list[dict[str, Any]]:
    """Freeze every budget-dependent adapter configuration into the manifest."""
    configurations: list[dict[str, Any]] = []
    for optimizer in optimizers:
        by_budget = []
        for budget in budgets:
            configuration, _ = _resolved_configuration(optimizer, budget)
            by_budget.append({"budget": budget, "configuration": configuration})
        configurations.append({"optimizer": optimizer.name, "by_budget": by_budget})
    return configurations


def _bounded_futures(
    executor: Executor,
    function: Callable[..., dict[str, Any]],
    work: list[WorkItem],
    max_in_flight: int,
) -> Iterator[tuple[Future[dict[str, Any]], WorkItem]]:
    """Submit work incrementally so large campaigns do not materialize every future."""
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1")

    work_iterator = iter(work)
    pending: dict[Future[dict[str, Any]], WorkItem] = {}

    def submit_next() -> bool:
        try:
            item = next(work_iterator)
        except StopIteration:
            return False
        pending[executor.submit(function, *item)] = item
        return True

    for _ in range(min(max_in_flight, len(work))):
        submit_next()

    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            item = pending.pop(future)
            yield future, item
            submit_next()


# ---------------------------------------------------------------------------
# Single-objective execution
# ---------------------------------------------------------------------------


def _run_single_one(
    problem: SingleObjectiveProblem,
    optimizer: SingleObjectiveOptimizer,
    budget: int,
    run_id: int,
) -> dict[str, Any]:
    """Execute one single-objective run. Must be top-level for pickling."""
    seed = make_seed(problem.name, budget, run_id)
    configuration, configuration_error = _resolved_configuration(optimizer, budget)
    row: dict[str, Any] = {
        "problem": problem.name,
        "optimizer": optimizer.name,
        "budget": budget,
        "run_id": run_id,
        "seed": seed,
        "status": "error",
        "error": "",
        "optimizer_config": configuration,
        "n_evaluations": None,
        "best_value": None,
        "best_params": None,
        "wall_time_seconds": None,
        "convergence_trace": None,
    }
    if configuration_error is not None:
        row["error"] = _format_error(configuration_error)
        return row

    started = time.perf_counter()
    try:
        result = optimizer.optimize(problem, budget, seed)
    except Exception as error:
        row["error"] = _format_error(error)
        row["wall_time_seconds"] = time.perf_counter() - started
        if isinstance(error, EvaluationCountError):
            row["n_evaluations"] = error.actual
        return row

    row.update(
        {
            "status": "success",
            "n_evaluations": result.n_evaluations,
            "best_value": result.best_value,
            "best_params": result.best_params,
            "wall_time_seconds": result.wall_time_seconds,
            "convergence_trace": result.convergence_trace,
        }
    )
    return row


def _single_executor_failure(
    problem: SingleObjectiveProblem,
    optimizer: SingleObjectiveOptimizer,
    budget: int,
    run_id: int,
    error: Exception,
) -> dict[str, Any]:
    """Create a row when a worker process fails outside the adapter call."""
    configuration, _ = _resolved_configuration(optimizer, budget)
    return {
        "problem": problem.name,
        "optimizer": optimizer.name,
        "budget": budget,
        "run_id": run_id,
        "seed": make_seed(problem.name, budget, run_id),
        "status": "error",
        "error": _format_error(error),
        "optimizer_config": configuration,
        "n_evaluations": None,
        "best_value": None,
        "best_params": None,
        "wall_time_seconds": None,
        "convergence_trace": None,
    }


def _print_single_progress(done: int, total: int, row: dict[str, Any]) -> None:
    outcome = f"{row['best_value']:.6f}" if row["status"] == "success" else f"ERROR: {row['error']}"
    print(
        f"  [{done}/{total}] {row['problem']} / {row['optimizer']} "
        f"/ budget={row['budget']} -> {outcome}"
    )


def run_single_objective(
    problems: list[SingleObjectiveProblem],
    optimizers: list[SingleObjectiveOptimizer],
    config: RunConfig,
) -> None:
    """Run all single-objective benchmarks."""
    manifest = build_campaign_manifest(
        run_kind="single_objective",
        budgets=config.budgets,
        n_runs=config.n_runs,
        problem_names=[problem.name for problem in problems],
        optimizer_names=[optimizer.name for optimizer in optimizers],
        optimizer_configurations=_campaign_configurations(optimizers, config.budgets),
    )
    store = ResultStore(config.output_dir)
    store.prepare_campaign(manifest, resume=config.resume)
    completed = store.completed_so_runs() if config.resume else set()

    # Build work queue
    work: list[tuple[SingleObjectiveProblem, SingleObjectiveOptimizer, int, int]] = []
    for problem in problems:
        for optimizer in optimizers:
            for budget in config.budgets:
                for run_id in range(config.n_runs):
                    key = (problem.name, optimizer.name, budget, run_id)
                    if key not in completed:
                        work.append((problem, optimizer, budget, run_id))

    total = len(work)
    if total == 0:
        print("All single-objective runs already completed.")
        return

    print(
        f"Running {total} single-objective evaluations "
        f"({len(problems)} problems x {len(optimizers)} optimizers)"
    )

    done = 0
    if config.effective_workers == 1:
        for problem, optimizer, budget, run_id in work:
            row = _run_single_one(problem, optimizer, budget, run_id)
            store.append_single(row)
            done += 1
            if done % 100 == 0 or done == total:
                _print_single_progress(done, total, row)
    else:
        with ProcessPoolExecutor(max_workers=config.effective_workers) as executor:
            for future, item in _bounded_futures(
                executor,
                _run_single_one,
                work,
                max_in_flight=2 * config.effective_workers,
            ):
                problem, optimizer, budget, run_id = item
                try:
                    row = future.result()
                except Exception as error:
                    row = _single_executor_failure(problem, optimizer, budget, run_id, error)
                store.append_single(row)
                done += 1
                if done % 100 == 0 or done == total:
                    _print_single_progress(done, total, row)


# ---------------------------------------------------------------------------
# Multi-objective execution
# ---------------------------------------------------------------------------


def _run_multi_one(
    problem: MultiObjectiveProblem,
    optimizer: MultiObjectiveOptimizer,
    budget: int,
    run_id: int,
) -> dict[str, Any]:
    """Execute one multi-objective run."""
    seed = make_seed(problem.name, budget, run_id)
    configuration, configuration_error = _resolved_configuration(optimizer, budget)
    row: dict[str, Any] = {
        "problem": problem.name,
        "optimizer": optimizer.name,
        "budget": budget,
        "run_id": run_id,
        "seed": seed,
        "status": "error",
        "error": "",
        "optimizer_config": configuration,
        "n_evaluations": None,
        "pareto_front": None,
        "decision_vectors": None,
        "normalized_hypervolume_gap": None,
        "normalized_igd": None,
        "spacing": None,
        "wall_time_seconds": None,
        "n_pareto_points": None,
    }
    if configuration_error is not None:
        row["error"] = _format_error(configuration_error)
        return row

    started = time.perf_counter()
    try:
        result = optimizer.optimize(problem, budget, seed)
        row.update(
            {
                "n_evaluations": result.n_evaluations,
                "pareto_front": result.pareto_front,
                "decision_vectors": result.decision_vectors,
                "wall_time_seconds": result.wall_time_seconds,
                "n_pareto_points": len(result.pareto_front),
            }
        )

        # Compute reporting metrics from the same raw objective front that is persisted.
        if problem.ideal_point is None:
            raise ValueError(f"{problem.name} does not define an objective ideal point")
        if problem.true_pareto_front is None:
            row["normalized_hypervolume_gap"] = float("nan")
            row["normalized_igd"] = float("nan")
        else:
            row["normalized_hypervolume_gap"] = compute_normalized_hv_gap(
                result.pareto_front,
                problem.true_pareto_front,
                problem.ideal_point,
                problem.reference_point,
                reference_hypervolume=problem.normalized_reference_hypervolume,
            )
            row["normalized_igd"] = compute_normalized_igd(
                result.pareto_front,
                problem.true_pareto_front,
                problem.ideal_point,
                problem.reference_point,
            )
        row["spacing"] = compute_normalized_spacing(
            result.pareto_front,
            problem.ideal_point,
            problem.reference_point,
        )
    except Exception as error:
        row["error"] = _format_error(error)
        if row["wall_time_seconds"] is None:
            row["wall_time_seconds"] = time.perf_counter() - started
        if isinstance(error, EvaluationCountError):
            row["n_evaluations"] = error.actual
        return row

    row["status"] = "success"
    return row


def _multi_executor_failure(
    problem: MultiObjectiveProblem,
    optimizer: MultiObjectiveOptimizer,
    budget: int,
    run_id: int,
    error: Exception,
) -> dict[str, Any]:
    """Create a row when a worker process fails outside the adapter call."""
    configuration, _ = _resolved_configuration(optimizer, budget)
    return {
        "problem": problem.name,
        "optimizer": optimizer.name,
        "budget": budget,
        "run_id": run_id,
        "seed": make_seed(problem.name, budget, run_id),
        "status": "error",
        "error": _format_error(error),
        "optimizer_config": configuration,
        "n_evaluations": None,
        "pareto_front": None,
        "decision_vectors": None,
        "normalized_hypervolume_gap": None,
        "normalized_igd": None,
        "spacing": None,
        "wall_time_seconds": None,
        "n_pareto_points": None,
    }


def _print_multi_progress(done: int, total: int, row: dict[str, Any]) -> None:
    outcome = (
        f"normalized HV gap={row['normalized_hypervolume_gap']:.4f}"
        if row["status"] == "success"
        else f"ERROR: {row['error']}"
    )
    print(
        f"  [{done}/{total}] {row['problem']} / {row['optimizer']} "
        f"/ budget={row['budget']} -> {outcome}"
    )


def run_multi_objective(
    problems: list[MultiObjectiveProblem],
    optimizers: list[MultiObjectiveOptimizer],
    config: RunConfig,
) -> None:
    """Run all multi-objective benchmarks."""
    manifest = build_campaign_manifest(
        run_kind="multi_objective",
        budgets=config.budgets,
        n_runs=config.n_runs,
        problem_names=[problem.name for problem in problems],
        optimizer_names=[optimizer.name for optimizer in optimizers],
        optimizer_configurations=_campaign_configurations(optimizers, config.budgets),
    )
    store = ResultStore(config.output_dir)
    store.prepare_campaign(manifest, resume=config.resume)
    completed = store.completed_mo_runs() if config.resume else set()

    work: list[tuple[MultiObjectiveProblem, MultiObjectiveOptimizer, int, int]] = []
    for problem in problems:
        for optimizer in optimizers:
            for budget in config.budgets:
                for run_id in range(config.n_runs):
                    key = (problem.name, optimizer.name, budget, run_id)
                    if key not in completed:
                        work.append((problem, optimizer, budget, run_id))

    total = len(work)
    if total == 0:
        print("All multi-objective runs already completed.")
        return

    print(
        f"Running {total} multi-objective evaluations "
        f"({len(problems)} problems x {len(optimizers)} optimizers)"
    )

    done = 0
    if config.effective_workers == 1:
        for problem, optimizer, budget, run_id in work:
            row = _run_multi_one(problem, optimizer, budget, run_id)
            store.append_multi(row)
            done += 1
            if done % 50 == 0 or done == total:
                _print_multi_progress(done, total, row)
    else:
        with ProcessPoolExecutor(max_workers=config.effective_workers) as executor:
            for future, item in _bounded_futures(
                executor,
                _run_multi_one,
                work,
                max_in_flight=2 * config.effective_workers,
            ):
                problem, optimizer, budget, run_id = item
                try:
                    row = future.result()
                except Exception as error:
                    row = _multi_executor_failure(problem, optimizer, budget, run_id, error)
                store.append_multi(row)
                done += 1
                if done % 50 == 0 or done == total:
                    _print_multi_progress(done, total, row)
