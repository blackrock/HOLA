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

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from benchmarks.adapters.base import (
    MultiObjectiveOptimizer,
    SingleObjectiveOptimizer,
)
from benchmarks.data.persistence import ResultStore
from benchmarks.metrics.hypervolume import compute_hv
from benchmarks.metrics.igd import compute_igd
from benchmarks.metrics.spacing import compute_spacing
from benchmarks.problems.registry import MultiObjectiveProblem, SingleObjectiveProblem
from benchmarks.runner.config import RunConfig


def _make_seed(problem_name: str, optimizer_name: str, budget: int, run_id: int) -> int:
    """Deterministic seed from run parameters."""
    key = f"{problem_name}:{optimizer_name}:{budget}:{run_id}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


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
    seed = _make_seed(problem.name, optimizer.name, budget, run_id)
    result = optimizer.optimize(problem, budget, seed)
    return {
        "problem": problem.name,
        "optimizer": optimizer.name,
        "budget": budget,
        "run_id": run_id,
        "seed": seed,
        "best_value": result.best_value,
        "wall_time_seconds": result.wall_time_seconds,
        "convergence_trace": result.convergence_trace,
    }


def run_single_objective(
    problems: list[SingleObjectiveProblem],
    optimizers: list[SingleObjectiveOptimizer],
    config: RunConfig,
) -> None:
    """Run all single-objective benchmarks."""
    store = ResultStore(config.output_dir)
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
                print(
                    f"  [{done}/{total}] {row['problem']} / {row['optimizer']} "
                    f"/ budget={row['budget']} -> {row['best_value']:.6f}"
                )
    else:
        with ProcessPoolExecutor(max_workers=config.effective_workers) as executor:
            futures = {
                executor.submit(_run_single_one, p, o, b, r): (p.name, o.name, b, r)
                for p, o, b, r in work
            }
            for future in as_completed(futures):
                row = future.result()
                store.append_single(row)
                done += 1
                if done % 100 == 0 or done == total:
                    print(
                        f"  [{done}/{total}] {row['problem']} / {row['optimizer']} "
                        f"/ budget={row['budget']} -> {row['best_value']:.6f}"
                    )


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
    seed = _make_seed(problem.name, optimizer.name, budget, run_id)
    result = optimizer.optimize(problem, budget, seed)

    # Compute metrics
    hv = compute_hv(result.pareto_front, problem.reference_point)
    igd_val = (
        compute_igd(result.pareto_front, problem.true_pareto_front)
        if problem.true_pareto_front is not None
        else float("nan")
    )
    spacing_val = compute_spacing(result.pareto_front)

    return {
        "problem": problem.name,
        "optimizer": optimizer.name,
        "budget": budget,
        "run_id": run_id,
        "seed": seed,
        "hypervolume": hv,
        "igd": igd_val,
        "spacing": spacing_val,
        "wall_time_seconds": result.wall_time_seconds,
        "n_pareto_points": len(result.pareto_front),
    }


def run_multi_objective(
    problems: list[MultiObjectiveProblem],
    optimizers: list[MultiObjectiveOptimizer],
    config: RunConfig,
) -> None:
    """Run all multi-objective benchmarks."""
    store = ResultStore(config.output_dir)
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
                print(
                    f"  [{done}/{total}] {row['problem']} / {row['optimizer']} "
                    f"/ budget={row['budget']} -> HV={row['hypervolume']:.4f}"
                )
    else:
        with ProcessPoolExecutor(max_workers=config.effective_workers) as executor:
            futures = {
                executor.submit(_run_multi_one, p, o, b, r): (p.name, o.name, b, r)
                for p, o, b, r in work
            }
            for future in as_completed(futures):
                row = future.result()
                store.append_multi(row)
                done += 1
                if done % 50 == 0 or done == total:
                    print(
                        f"  [{done}/{total}] {row['problem']} / {row['optimizer']} "
                        f"/ budget={row['budget']} -> HV={row['hypervolume']:.4f}"
                    )
