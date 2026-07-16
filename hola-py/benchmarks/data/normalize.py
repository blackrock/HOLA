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

"""Problem-defined single-objective regret and summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from benchmarks.problems.registry import SingleObjectiveProblem

REGRET_TOLERANCE = 1e-9
CALIBRATION_OPTIMIZERS = frozenset({"Random x2"})
GMM_OPTIMIZER = "HOLA (GMM)"
GMM_MECHANISM_COMPARATORS = (
    "HOLA (random)",
    "HOLA (sobol)",
    "Random x2",
)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260715
DEFAULT_CONFIDENCE_LEVEL = 0.95


def paired_win_outcomes(
    df: pd.DataFrame,
    *,
    focal_optimizer: str = GMM_OPTIMIZER,
    comparators: Sequence[str] = GMM_MECHANISM_COMPARATORS,
) -> pd.DataFrame:
    """Return paired win, tie, and loss scores for one focal optimizer.

    Successful pairs compare simple regret exactly. A focal-only success is a
    win, a comparator-only success is a loss, and two failed runs tie. Pairing
    is by task, budget, and run ID, never by input row order.
    """
    required = {
        "suite",
        "family",
        "problem",
        "dimension",
        "optimizer",
        "budget",
        "run_id",
        "status",
        "regret",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing paired-comparison columns: {', '.join(sorted(missing))}")
    if not comparators:
        raise ValueError("paired comparison requires at least one comparator")
    if focal_optimizer in comparators or len(comparators) != len(set(comparators)):
        raise ValueError("paired comparison optimizer names must be distinct")

    selected_names = {focal_optimizer, *comparators}
    selected = df[df["optimizer"].isin(selected_names)].copy()
    invalid_status = selected["status"].isna() | ~selected["status"].isin({"success", "error"})
    if invalid_status.any():
        raise ValueError("paired comparison contains invalid result statuses")

    key_columns = [
        "suite",
        "family",
        "problem",
        "dimension",
        "budget",
        "run_id",
    ]
    duplicate = selected.duplicated([*key_columns, "optimizer"], keep=False)
    if duplicate.any():
        raise ValueError("paired comparison contains duplicate optimizer run keys")

    focal = selected[selected["optimizer"].eq(focal_optimizer)][
        [*key_columns, "status", "regret"]
    ].rename(columns={"status": "focal_status", "regret": "focal_regret"})
    outcome_frames: list[pd.DataFrame] = []
    for comparator in comparators:
        other = selected[selected["optimizer"].eq(comparator)][
            [*key_columns, "status", "regret"]
        ].rename(columns={"status": "comparator_status", "regret": "comparator_regret"})
        paired = focal.merge(
            other,
            on=key_columns,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError(f"unpaired runs between {focal_optimizer!r} and {comparator!r}")
        paired = paired.drop(columns="_merge")
        focal_success = paired["focal_status"].eq("success")
        comparator_success = paired["comparator_status"].eq("success")
        successful_pair = focal_success & comparator_success
        invalid_metric = successful_pair & (
            ~np.isfinite(pd.to_numeric(paired["focal_regret"], errors="coerce"))
            | ~np.isfinite(pd.to_numeric(paired["comparator_regret"], errors="coerce"))
        )
        if invalid_metric.any():
            raise ValueError("successful paired runs require finite simple regret")

        outcome = np.full(len(paired), 0.5, dtype=float)
        outcome[focal_success & ~comparator_success] = 1.0
        outcome[~focal_success & comparator_success] = 0.0
        focal_regret = pd.to_numeric(paired["focal_regret"], errors="coerce").to_numpy()
        comparator_regret = pd.to_numeric(paired["comparator_regret"], errors="coerce").to_numpy()
        outcome[successful_pair & (focal_regret < comparator_regret)] = 1.0
        outcome[successful_pair & (focal_regret > comparator_regret)] = 0.0
        paired["focal_optimizer"] = focal_optimizer
        paired["comparator"] = comparator
        paired["outcome"] = outcome
        outcome_frames.append(paired)

    return (
        pd.concat(outcome_frames, ignore_index=True)
        .sort_values(["suite", "budget", "comparator", "family", "problem", "run_id"])
        .reset_index(drop=True)
    )


def aggregate_family_balanced_paired_win_rates(
    df: pd.DataFrame,
    *,
    focal_optimizer: str = GMM_OPTIMIZER,
    comparators: Sequence[str] = GMM_MECHANISM_COMPARATORS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Summarize paired outcomes with equal weight per dimension and family.

    For each bootstrap replicate, we resample paired outcome scores
    independently within each concrete task, average dimensional variants
    within each named family, and then average families. Run IDs preserve the
    focal/comparator pairing within a task; the same label in unrelated tasks
    does not define a bootstrap cluster. This prevents Ackley, Rastrigin, and
    Schwefel from receiving extra weight merely because the suite includes
    several dimensions.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    outcomes = paired_win_outcomes(
        df,
        focal_optimizer=focal_optimizer,
        comparators=comparators,
    )
    rows: list[dict[str, object]] = []
    group_columns = ["suite", "budget", "comparator"]
    alpha = 1.0 - confidence_level
    for group_index, (key, group) in enumerate(outcomes.groupby(group_columns, sort=True)):
        suite, budget, comparator = cast(tuple[str, int, str], key)
        task_columns = ["family", "problem", "dimension"]
        run_sets = group.groupby(task_columns, sort=True)["run_id"].agg(
            lambda values: tuple(sorted(values))
        )
        if run_sets.empty or run_sets.nunique() != 1:
            raise ValueError(
                f"paired bootstrap requires identical run IDs for every task in {suite!r}"
            )

        family_run = group.groupby(["family", "run_id"], sort=True)["outcome"].mean().reset_index()
        per_run = family_run.groupby("run_id", sort=True)["outcome"].mean()
        run_scores = per_run.to_numpy(dtype=float)
        rng = np.random.default_rng(np.random.SeedSequence([seed, group_index]))
        samples = np.zeros(n_bootstrap, dtype=float)
        for _, family_group in group.groupby("family", sort=True):
            family_samples = np.zeros(n_bootstrap, dtype=float)
            n_family_tasks = 0
            for _, task_group in family_group.groupby(["problem", "dimension"], sort=True):
                # Outcome order must not give arbitrary run-ID labels meaning
                # in the deterministic bootstrap stream.
                task_outcomes = np.sort(task_group["outcome"].to_numpy(dtype=float))
                family_samples += rng.choice(
                    task_outcomes,
                    size=(n_bootstrap, len(task_outcomes)),
                    replace=True,
                ).mean(axis=1)
                n_family_tasks += 1
            samples += family_samples / n_family_tasks
        samples /= group["family"].nunique()
        lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        rows.append(
            {
                "suite": suite,
                "budget": budget,
                "focal_optimizer": focal_optimizer,
                "comparator": comparator,
                "win_rate": float(run_scores.mean()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_families": int(group["family"].nunique()),
                "n_tasks": int(group[task_columns].drop_duplicates().shape[0]),
                "n_paired_runs": len(run_scores),
                "n_paired_outcomes": len(group),
                "n_wins": int(group["outcome"].eq(1.0).sum()),
                "n_ties": int(group["outcome"].eq(0.5).sum()),
                "n_losses": int(group["outcome"].eq(0.0).sum()),
                "confidence_level": confidence_level,
                "bootstrap_samples": n_bootstrap,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows).sort_values(["suite", "comparator", "budget"])


def lexicographic_failure_ranks(
    failure_counts: pd.Series,
    metric_values: pd.Series,
) -> pd.Series:
    """Rank configurations by failures first, then a lower-is-better metric.

    The comparison key is ``(number of failed runs, median successful-run
    metric)``. Missing or non-finite metrics sort last among configurations
    with the same failure count, and configurations with identical keys share
    the average of their occupied ranks.
    """
    if not failure_counts.index.equals(metric_values.index):
        raise ValueError("failure counts and metric values must have identical indexes")
    if failure_counts.index.has_duplicates:
        raise ValueError("failure-aware ranking requires unique configuration indexes")

    numeric_failures = pd.to_numeric(failure_counts, errors="coerce")
    invalid_failures = numeric_failures.isna() | (numeric_failures < 0)
    invalid_failures |= numeric_failures.mod(1).ne(0)
    if invalid_failures.any():
        raise ValueError("failure counts must be non-negative integers")

    numeric_metrics = pd.to_numeric(metric_values, errors="coerce")
    keys = {
        optimizer: (
            int(numeric_failures.loc[optimizer]),
            (
                float(numeric_metrics.loc[optimizer])
                if math.isfinite(float(numeric_metrics.loc[optimizer]))
                else float("inf")
            ),
        )
        for optimizer in numeric_failures.index
    }
    ordered = sorted(keys, key=lambda optimizer: keys[optimizer])
    ranks = pd.Series(index=failure_counts.index, dtype=float)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and keys[ordered[end]] == keys[ordered[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for optimizer in ordered[start:end]:
            ranks.loc[optimizer] = average_rank
        start = end
    return ranks


def add_simple_regret(
    df: pd.DataFrame,
    problems: Mapping[str, SingleObjectiveProblem],
    *,
    tolerance: float = REGRET_TOLERANCE,
) -> pd.DataFrame:
    """Attach fixed known-optimum regret and problem metadata to result rows.

    Regret is ``best_value - known_minimum``. Tiny negative values within the
    numerical tolerance are set to zero; a larger negative value raises so a
    bad registry optimum or objective implementation cannot be hidden.
    """
    required = {"problem", "best_value"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"missing result columns: {', '.join(sorted(missing_columns))}")
    if tolerance < 0.0:
        raise ValueError("regret tolerance must be non-negative")

    unknown = sorted(set(df["problem"].dropna()) - set(problems))
    if unknown:
        raise ValueError(f"unknown benchmark problems: {', '.join(unknown)}")

    minima = {name: problem.known_minimum for name, problem in problems.items()}
    families = {name: problem.family or problem.name for name, problem in problems.items()}
    suites = {name: problem.suite for name, problem in problems.items()}
    dimensions = {name: problem.dimensionality for name, problem in problems.items()}

    result = df.copy()
    result["known_minimum"] = result["problem"].map(minima)
    values = pd.to_numeric(result["best_value"], errors="coerce")
    if "status" in result:
        values = values.where(result["status"].eq("success"))
    regret = values - result["known_minimum"]
    invalid = regret < -tolerance
    if invalid.any():
        row = result.loc[invalid, ["problem", "best_value", "known_minimum"]].iloc[0]
        raise ValueError(
            f"{row['problem']} result {row['best_value']} is below registered optimum "
            f"{row['known_minimum']} by more than {tolerance}"
        )
    result["regret"] = regret.mask((regret < 0.0) & ~invalid, 0.0)
    result["family"] = result["problem"].map(families)
    result["suite"] = result["problem"].map(suites)
    result["dimension"] = result["problem"].map(dimensions)
    return result


def summarize_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize successful-run regret and explicit outcome counts by task."""
    required = {
        "suite",
        "family",
        "problem",
        "dimension",
        "optimizer",
        "budget",
        "status",
        "regret",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"missing regret columns: {', '.join(sorted(missing_columns))}")

    values = df.copy()
    invalid_statuses = sorted(set(values["status"].dropna()) - {"success", "error"})
    if invalid_statuses or values["status"].isna().any():
        rendered = ", ".join(map(str, invalid_statuses)) or "missing"
        raise ValueError(f"invalid benchmark result statuses: {rendered}")
    values["_successful"] = values["status"].eq("success")
    values["_failed"] = values["status"].eq("error")
    values["regret"] = pd.to_numeric(values["regret"], errors="coerce").where(values["_successful"])
    if "n_evaluations" not in values:
        values["n_evaluations"] = values["budget"]
    values["n_evaluations"] = pd.to_numeric(values["n_evaluations"], errors="coerce").where(
        values["_successful"]
    )
    group_columns = ["suite", "family", "problem", "dimension", "optimizer", "budget"]
    return (
        values.groupby(group_columns, dropna=False)
        .agg(
            regret_median=("regret", "median"),
            regret_q1=("regret", lambda values: values.quantile(0.25)),
            regret_q3=("regret", lambda values: values.quantile(0.75)),
            n_runs=("status", "size"),
            n_total_runs=("status", "size"),
            n_successful_runs=("_successful", "sum"),
            n_failed_runs=("_failed", "sum"),
            actual_evaluations_median=("n_evaluations", "median"),
            actual_evaluations_min=("n_evaluations", "min"),
            actual_evaluations_max=("n_evaluations", "max"),
        )
        .reset_index()
        .assign(success_rate=lambda summary: summary["n_successful_runs"] / summary["n_total_runs"])
        .sort_values(["suite", "family", "dimension", "budget", "regret_median"])
    )


def aggregate_family_balanced_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Average task ranks with each named function family weighted once.

    Concrete dimensional variants are ranked separately and averaged within
    their family before family ranks are averaged. Budgets and suites remain
    separate, so the practical task is never pooled with synthetic functions.
    Each task is ranked lexicographically by fewer failed runs and then lower
    median regret among successful runs. Exact ties share their average rank.
    """
    required = {
        "suite",
        "family",
        "problem",
        "dimension",
        "optimizer",
        "budget",
        "regret_median",
        "n_failed_runs",
    }
    missing_columns = required - set(summary.columns)
    if missing_columns:
        raise ValueError(f"missing summary columns: {', '.join(sorted(missing_columns))}")

    primary = summary[~summary["optimizer"].isin(CALIBRATION_OPTIMIZERS)].copy()
    optimizer_sets: dict[tuple[str, int], list[str]] = {}
    for key, group in primary.groupby(["suite", "budget"]):
        suite, budget = cast(tuple[str, int], key)
        optimizer_sets[(suite, budget)] = sorted(group["optimizer"].unique())
    rank_rows: list[dict[str, object]] = []
    task_columns = ["suite", "budget", "family", "problem", "dimension"]
    for task, group in primary.groupby(task_columns):
        suite, budget, family, problem, dimension = cast(tuple[str, int, str, str, int], task)
        optimizers = optimizer_sets[(suite, budget)]
        indexed = group.set_index("optimizer").reindex(optimizers)
        ranks = lexicographic_failure_ranks(
            indexed["n_failed_runs"],
            indexed["regret_median"],
        )
        for optimizer, rank in ranks.items():
            rank_rows.append(
                {
                    "suite": suite,
                    "budget": budget,
                    "family": family,
                    "problem": problem,
                    "dimension": dimension,
                    "optimizer": optimizer,
                    "task_rank": float(rank),
                }
            )

    task_ranks = pd.DataFrame(rank_rows)
    if task_ranks.empty:
        return pd.DataFrame(columns=["suite", "budget", "optimizer", "mean_rank", "n_families"])
    family_ranks = (
        task_ranks.groupby(["suite", "budget", "optimizer", "family"])["task_rank"]
        .mean()
        .reset_index(name="family_rank")
    )
    return (
        family_ranks.groupby(["suite", "budget", "optimizer"])
        .agg(mean_rank=("family_rank", "mean"), n_families=("family", "nunique"))
        .reset_index()
        .sort_values(["suite", "budget", "mean_rank"])
    )
