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

"""Multi-objective plots for fixed-scale hypervolume gap and IGD."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.data.normalize import lexicographic_failure_ranks
from benchmarks.data.persistence import ResultStore
from benchmarks.plotting.export import save_figure
from benchmarks.plotting.style import apply_paper_style, get_color
from benchmarks.problems.multi_objective import MULTI_OBJECTIVE_PROBLEMS

plt.switch_backend("Agg")

PRIMARY_METRICS = {
    "normalized_hypervolume_gap": "Normalized hypervolume gap",
    "normalized_igd": "Normalized IGD",
}
FAILURE_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "run_id",
    "seed",
    "error",
    "n_evaluations",
]
REPRESENTATIVE_FRONT_PROBLEMS = ("zdt3_30d", "wfg1_2obj_24d")
REPRESENTATIVE_SELECTION_RULE = (
    "among successful runs with a finite normalized hypervolume gap at the task's "
    "largest budget, minimum absolute distance to the optimizer/task/budget median "
    "hypervolume gap; ties by minimum absolute distance to the median finite IGD, "
    "then lower run_id"
)
REPRESENTATIVE_SELECTION_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "selection_status",
    "source_result_row_index",
    "run_id",
    "seed",
    "result_status",
    "error",
    "n_evaluations",
    "wall_time_seconds",
    "normalized_hypervolume_gap",
    "method_median_hypervolume_gap",
    "absolute_distance_to_median",
    "normalized_igd",
    "method_median_igd",
    "absolute_igd_distance_to_median",
    "spacing",
    "stored_n_pareto_points",
    "parsed_n_pareto_points",
    "n_runs_at_budget",
    "n_successful_runs_at_budget",
    "n_successful_finite_hv_runs_at_budget",
    "n_failed_runs_at_budget",
    "failure_errors",
    "front_validation_error",
    "selection_rule",
]
REPRESENTATIVE_POINT_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "run_id",
    "seed",
    "point_index",
    "objective_1",
    "objective_2",
]


def parse_stored_objective_front(value: Any, n_objectives: int) -> np.ndarray:
    """Parse an untrusted stored objective front without executable decoding."""
    if n_objectives <= 0:
        raise ValueError("n_objectives must be positive")
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("stored Pareto front is not valid JSON") from error
    try:
        front = np.asarray(parsed, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("stored Pareto front must contain only numeric values") from error
    if front.ndim != 2 or front.shape[1] != n_objectives:
        raise ValueError(f"stored Pareto front must have shape (n, {n_objectives})")
    if len(front) == 0:
        raise ValueError("stored Pareto front must be non-empty")
    if not np.all(np.isfinite(front)):
        raise ValueError("stored Pareto front must contain only finite values")
    return front


def _optional_numeric(row: pd.Series, column: str) -> float | None:
    if column not in row:
        return None
    numeric = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _failure_errors(rows: pd.DataFrame) -> str:
    if "error" not in rows:
        return ""
    errors = {
        str(value).strip()
        for value in rows.loc[rows["status"].eq("error"), "error"]
        if pd.notna(value) and str(value).strip()
    }
    return " | ".join(sorted(errors))


def _empty_representative_selection(
    problem: str,
    optimizer: str,
    budget: int,
    status: str,
    optimizer_rows: pd.DataFrame,
) -> dict[str, object]:
    successful = optimizer_rows["status"].eq("success")
    failed = optimizer_rows["status"].eq("error")
    return {
        "problem": problem,
        "optimizer": optimizer,
        "budget": budget,
        "selection_status": status,
        "source_result_row_index": None,
        "run_id": None,
        "seed": None,
        "result_status": None,
        "error": None,
        "n_evaluations": None,
        "wall_time_seconds": None,
        "normalized_hypervolume_gap": None,
        "method_median_hypervolume_gap": None,
        "absolute_distance_to_median": None,
        "normalized_igd": None,
        "method_median_igd": None,
        "absolute_igd_distance_to_median": None,
        "spacing": None,
        "stored_n_pareto_points": None,
        "parsed_n_pareto_points": None,
        "n_runs_at_budget": len(optimizer_rows),
        "n_successful_runs_at_budget": int(successful.sum()),
        "n_successful_finite_hv_runs_at_budget": 0,
        "n_failed_runs_at_budget": int(failed.sum()),
        "failure_errors": _failure_errors(optimizer_rows),
        "front_validation_error": None,
        "selection_rule": REPRESENTATIVE_SELECTION_RULE,
    }


def representative_terminal_front_tables(
    results: pd.DataFrame,
    problem_names: Sequence[str] = REPRESENTATIVE_FRONT_PROBLEMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one auditable terminal front per optimizer and requested task."""
    required = {
        "problem",
        "optimizer",
        "budget",
        "run_id",
        "seed",
        "status",
        "pareto_front",
        "normalized_hypervolume_gap",
        "normalized_igd",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(
            "multi-objective results lack representative-front columns: "
            + ", ".join(sorted(missing))
        )
    unknown = sorted(set(problem_names) - set(MULTI_OBJECTIVE_PROBLEMS))
    if unknown:
        raise ValueError(f"unknown representative-front problems: {', '.join(unknown)}")

    values = results.copy()
    values["_source_result_row_index"] = np.arange(len(values), dtype=int)
    optimizers = sorted(str(value) for value in values["optimizer"].dropna().unique())
    selection_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    for problem_name in problem_names:
        problem_rows = values[values["problem"].eq(problem_name)]
        if problem_rows.empty:
            continue
        problem = MULTI_OBJECTIVE_PROBLEMS[problem_name]
        if problem.n_objectives != 2:
            raise ValueError("representative terminal-front plots require two-objective tasks")
        largest_budget = int(pd.to_numeric(problem_rows["budget"], errors="raise").max())
        largest = problem_rows[problem_rows["budget"].eq(largest_budget)]

        for optimizer in optimizers:
            optimizer_rows = largest[largest["optimizer"].eq(optimizer)]
            if optimizer_rows.empty:
                selection_rows.append(
                    _empty_representative_selection(
                        problem_name,
                        optimizer,
                        largest_budget,
                        "no_result_at_largest_budget",
                        optimizer_rows,
                    )
                )
                continue

            metric = pd.to_numeric(optimizer_rows["normalized_hypervolume_gap"], errors="coerce")
            finite_metric = pd.Series(
                np.isfinite(metric.to_numpy(dtype=float, na_value=np.nan)),
                index=optimizer_rows.index,
            )
            candidates = optimizer_rows[
                optimizer_rows["status"].eq("success") & finite_metric
            ].copy()
            if candidates.empty:
                selection_rows.append(
                    _empty_representative_selection(
                        problem_name,
                        optimizer,
                        largest_budget,
                        "no_successful_finite_hv_run",
                        optimizer_rows,
                    )
                )
                continue

            candidates["normalized_hypervolume_gap"] = pd.to_numeric(
                candidates["normalized_hypervolume_gap"], errors="raise"
            )
            median_gap = float(candidates["normalized_hypervolume_gap"].median())
            candidates["_distance_to_median"] = (
                candidates["normalized_hypervolume_gap"] - median_gap
            ).abs()
            minimum_distance = float(candidates["_distance_to_median"].min())
            tied = candidates[
                np.isclose(
                    candidates["_distance_to_median"],
                    minimum_distance,
                    rtol=0.0,
                    atol=1e-15,
                )
            ].copy()
            candidate_igd = pd.to_numeric(candidates["normalized_igd"], errors="coerce")
            finite_candidate_igd = np.isfinite(candidate_igd.to_numpy(dtype=float))
            median_igd = (
                float(candidate_igd.loc[finite_candidate_igd].median())
                if finite_candidate_igd.any()
                else None
            )
            tied["_igd"] = pd.to_numeric(tied["normalized_igd"], errors="coerce")
            tied["_igd_distance_to_median"] = (
                (tied["_igd"] - median_igd).abs() if median_igd is not None else np.nan
            )
            finite_tied_igd = np.isfinite(tied["_igd_distance_to_median"].to_numpy(dtype=float))
            if finite_tied_igd.any():
                minimum_igd_distance = float(
                    tied.loc[finite_tied_igd, "_igd_distance_to_median"].min()
                )
                tied = tied[
                    np.isclose(
                        tied["_igd_distance_to_median"],
                        minimum_igd_distance,
                        rtol=0.0,
                        atol=1e-15,
                    )
                ].copy()
            tied["_run_id_sort"] = pd.to_numeric(tied["run_id"], errors="raise")
            chosen = tied.sort_values("_run_id_sort", kind="mergesort").iloc[0]

            successful = optimizer_rows["status"].eq("success")
            failed = optimizer_rows["status"].eq("error")
            selection_status = "selected"
            validation_error: str | None = None
            front: np.ndarray | None = None
            parsed_point_count: int | None = None
            try:
                front = parse_stored_objective_front(chosen["pareto_front"], problem.n_objectives)
                parsed_point_count = len(front)
                stored_count = _optional_numeric(chosen, "n_pareto_points")
                if stored_count is not None and (
                    not float(stored_count).is_integer() or int(stored_count) != len(front)
                ):
                    raise ValueError(
                        "stored n_pareto_points does not match the parsed Pareto front"
                    )
            except ValueError as error:
                selection_status = "invalid_stored_front"
                validation_error = str(error)
                front = None

            run_id = int(chosen["run_id"])
            seed = int(chosen["seed"])
            selected_error = (
                chosen["error"] if "error" in chosen and pd.notna(chosen["error"]) else None
            )
            selection_rows.append(
                {
                    "problem": problem_name,
                    "optimizer": optimizer,
                    "budget": largest_budget,
                    "selection_status": selection_status,
                    "source_result_row_index": int(chosen["_source_result_row_index"]),
                    "run_id": run_id,
                    "seed": seed,
                    "result_status": str(chosen["status"]),
                    "error": selected_error,
                    "n_evaluations": _optional_numeric(chosen, "n_evaluations"),
                    "wall_time_seconds": _optional_numeric(chosen, "wall_time_seconds"),
                    "normalized_hypervolume_gap": float(chosen["normalized_hypervolume_gap"]),
                    "method_median_hypervolume_gap": median_gap,
                    "absolute_distance_to_median": float(chosen["_distance_to_median"]),
                    "normalized_igd": _optional_numeric(chosen, "normalized_igd"),
                    "method_median_igd": median_igd,
                    "absolute_igd_distance_to_median": _optional_numeric(
                        chosen, "_igd_distance_to_median"
                    ),
                    "spacing": _optional_numeric(chosen, "spacing"),
                    "stored_n_pareto_points": _optional_numeric(chosen, "n_pareto_points"),
                    "parsed_n_pareto_points": parsed_point_count,
                    "n_runs_at_budget": len(optimizer_rows),
                    "n_successful_runs_at_budget": int(successful.sum()),
                    "n_successful_finite_hv_runs_at_budget": len(candidates),
                    "n_failed_runs_at_budget": int(failed.sum()),
                    "failure_errors": _failure_errors(optimizer_rows),
                    "front_validation_error": validation_error,
                    "selection_rule": REPRESENTATIVE_SELECTION_RULE,
                }
            )
            if front is None:
                continue
            for point_index, point in enumerate(front):
                point_rows.append(
                    {
                        "problem": problem_name,
                        "optimizer": optimizer,
                        "budget": largest_budget,
                        "run_id": run_id,
                        "seed": seed,
                        "point_index": point_index,
                        "objective_1": float(point[0]),
                        "objective_2": float(point[1]),
                    }
                )

    return (
        pd.DataFrame(selection_rows, columns=REPRESENTATIVE_SELECTION_COLUMNS),
        pd.DataFrame(point_rows, columns=REPRESENTATIVE_POINT_COLUMNS),
    )


def _representative_panel_notes(selections: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    for _, row in selections.sort_values("optimizer").iterrows():
        optimizer = str(row["optimizer"])
        status = str(row["selection_status"])
        failed = int(row["n_failed_runs_at_budget"])
        failed_suffix = f"; {failed} failed run{'s' if failed != 1 else ''}" if failed else ""
        if status == "selected":
            if failed:
                notes.append(f"{optimizer}: {failed} failed run{'s' if failed != 1 else ''}")
        elif status == "invalid_stored_front":
            notes.append(f"{optimizer}: selected front invalid{failed_suffix}")
        elif status == "no_successful_finite_hv_run":
            notes.append(f"{optimizer}: no successful finite-HV run{failed_suffix}")
        else:
            notes.append(f"{optimizer}: no result at largest budget")
    return notes


def plot_representative_terminal_fronts(
    selections: pd.DataFrame,
    points: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot selected terminal fronts against fixed true fronts and reference points."""
    if selections.empty:
        return
    apply_paper_style()
    problem_set = set(selections["problem"])
    problem_names = [name for name in REPRESENTATIVE_FRONT_PROBLEMS if name in problem_set]
    problem_names.extend(sorted(problem_set - set(problem_names)))
    fig, axes = plt.subplots(
        1,
        len(problem_names),
        figsize=(5.1 * len(problem_names), 4.1),
        layout="constrained",
        squeeze=False,
    )
    legend_entries: dict[str, Any] = {}

    for ax, problem_name in zip(axes[0], problem_names, strict=True):
        problem = MULTI_OBJECTIVE_PROBLEMS[problem_name]
        problem_selections = selections[selections["problem"].eq(problem_name)]
        budgets = problem_selections["budget"].unique()
        if len(budgets) != 1:
            raise ValueError("representative selections must use one budget per task")
        budget = int(budgets[0])
        if problem.true_pareto_front is None:
            raise ValueError(f"{problem_name} lacks a fixed true Pareto front")
        true_front = parse_stored_objective_front(problem.true_pareto_front, problem.n_objectives)
        reference = np.asarray(problem.reference_point, dtype=float)
        if reference.shape != (2,) or not np.all(np.isfinite(reference)):
            raise ValueError(f"{problem_name} lacks a finite two-objective reference point")

        ax.scatter(
            true_front[:, 0],
            true_front[:, 1],
            s=8,
            marker=".",
            linewidths=0,
            color="black",
            alpha=0.8,
            label="Fixed true Pareto front",
            zorder=1,
        )
        ax.scatter(
            [reference[0]],
            [reference[1]],
            s=42,
            marker="x",
            linewidths=1.5,
            color="black",
            label="Reporting reference point",
            zorder=3,
        )
        problem_points = points[points["problem"].eq(problem_name)]
        for optimizer_key, optimizer_points in problem_points.groupby("optimizer", sort=True):
            optimizer = str(optimizer_key)
            ax.scatter(
                optimizer_points["objective_1"],
                optimizer_points["objective_2"],
                s=16,
                linewidths=0,
                alpha=0.7,
                color=get_color(optimizer),
                label=optimizer,
                zorder=2,
            )

        notes = _representative_panel_notes(problem_selections)
        if notes:
            ax.text(
                0.02,
                0.02,
                "\n".join(notes),
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=6.5,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                zorder=4,
            )
        objective_1, objective_2 = problem.objective_names
        ax.set_xlabel(objective_1)
        ax.set_ylabel(objective_2)
        ax.set_title(f"{problem.family.upper()}, {budget} evaluations")
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles, strict=True))

    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="outside lower center",
        ncol=min(4, len(legend_entries)),
        fontsize=7,
    )
    save_figure(fig, output_dir, "multi_objective_representative_terminal_fronts")
    plt.close(fig)


def aggregate_family_balanced_metric_ranks(
    summary: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Failure-aware task ranks with every named problem family weighted once.

    Within each task, configurations are compared lexicographically by fewer
    failed runs and then lower median successful-run metric. Exact ties share
    their average rank.
    """
    required = {
        "problem",
        "family",
        "n_objectives",
        "optimizer",
        "budget",
        metric,
        "n_failed_runs",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"missing MO summary columns: {', '.join(sorted(missing))}")

    optimizer_sets: dict[int, list[str]] = {}
    for budget_key, group in summary.groupby("budget"):
        budget = cast(int, budget_key)
        optimizer_sets[budget] = sorted(group["optimizer"].unique())
    rank_rows: list[dict[str, object]] = []
    task_columns = ["budget", "family", "problem", "n_objectives"]
    for key, group in summary.groupby(task_columns):
        budget, family, problem, n_objectives = cast(tuple[int, str, str, int], key)
        optimizers = optimizer_sets[budget]
        indexed = group.set_index("optimizer").reindex(optimizers)
        ranks = lexicographic_failure_ranks(
            indexed["n_failed_runs"],
            indexed[metric],
        )
        for optimizer, rank in ranks.items():
            rank_rows.append(
                {
                    "budget": budget,
                    "family": family,
                    "problem": problem,
                    "n_objectives": n_objectives,
                    "optimizer": optimizer,
                    "task_rank": float(rank),
                }
            )

    task_ranks = pd.DataFrame(rank_rows)
    if task_ranks.empty:
        return pd.DataFrame(columns=["budget", "optimizer", "mean_rank", "n_families"])
    family_ranks = (
        task_ranks.groupby(["budget", "optimizer", "family"])["task_rank"]
        .mean()
        .reset_index(name="family_rank")
    )
    return (
        family_ranks.groupby(["budget", "optimizer"])
        .agg(mean_rank=("family_rank", "mean"), n_families=("family", "nunique"))
        .reset_index()
        .sort_values(["budget", "mean_rank"])
    )


def plot_family_balanced_metric_ranks(
    ranks: pd.DataFrame,
    output_dir: Path,
    metric: str,
    label: str,
) -> None:
    """Plot a compact family-balanced rank for one primary metric."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(ranks["budget"].unique())
    optimizers = sorted(ranks["optimizer"].unique())
    width = 0.8 / len(optimizers)
    fig, ax = plt.subplots(figsize=(6.3, 4), layout="constrained")
    for index, optimizer in enumerate(optimizers):
        rows = ranks[ranks["optimizer"] == optimizer].set_index("budget")
        present_budgets = [budget for budget in budgets if budget in rows.index]
        positions = [budgets.index(budget) + index * width for budget in present_budgets]
        ax.bar(
            positions,
            rows.loc[present_budgets, "mean_rank"],
            width=width,
            label=optimizer,
            color=get_color(optimizer),
        )

    ax.set_xticks([position + width * len(optimizers) / 2 for position in range(len(budgets))])
    ax.set_xticklabels(budgets)
    ax.set_xlabel("Completed objective evaluations")
    ax.set_ylabel("Mean family-balanced rank (lower is better)")
    ax.set_title(f"Family-balanced {label} rank")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        borderaxespad=0,
    )
    save_figure(fig, output_dir, f"family_balanced_rank_{metric}")
    plt.close(fig)


def plot_metric_by_budget(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    label: str,
) -> None:
    """Plot a lower-is-better normalized metric separately for each problem."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    for problem_name, group in df.groupby("problem"):
        name = str(problem_name)
        fig, ax = plt.subplots(figsize=(6.3, 3.5), layout="constrained")
        budgets = sorted(group["budget"].unique())
        optimizers = sorted(group["optimizer"].unique())
        width = 0.8 / len(optimizers)

        for index, optimizer in enumerate(optimizers):
            optimizer_data = group[group["optimizer"] == optimizer]
            medians = [
                optimizer_data[optimizer_data["budget"] == budget][metric].median()
                for budget in budgets
            ]
            positions = [position + index * width for position in range(len(budgets))]
            ax.bar(positions, medians, width=width, label=optimizer, color=get_color(optimizer))

        ax.set_xticks([position + width * len(optimizers) / 2 for position in range(len(budgets))])
        ax.set_xticklabels(budgets)
        ax.set_xlabel("Completed objective evaluations")
        ax.set_ylabel(f"{label} (lower is better)")
        ax.set_title(name.replace("_", " "))
        ax.legend(
            fontsize=7,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

        save_figure(fig, output_dir, f"{metric}_{name}")
        plt.close(fig)

    print(f"Saved {label} plots to {output_dir}")


def summarize_multiobjective_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize successful-run metrics and explicit outcomes by task."""
    required = {
        "problem",
        "optimizer",
        "budget",
        "status",
        *PRIMARY_METRICS,
        "spacing",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "multi-objective results lack fixed-scale reporting columns: "
            + ", ".join(sorted(missing))
        )
    unknown = sorted(set(df["problem"].dropna()) - set(MULTI_OBJECTIVE_PROBLEMS))
    if unknown:
        raise ValueError(f"unknown multi-objective problems: {', '.join(unknown)}")
    values = df.copy()
    invalid_statuses = sorted(set(values["status"].dropna()) - {"success", "error"})
    if invalid_statuses or values["status"].isna().any():
        rendered = ", ".join(map(str, invalid_statuses)) or "missing"
        raise ValueError(f"invalid benchmark result statuses: {rendered}")
    values["_successful"] = values["status"].eq("success")
    values["_failed"] = values["status"].eq("error")
    for metric in (*PRIMARY_METRICS, "spacing"):
        values[metric] = pd.to_numeric(values[metric], errors="coerce").where(values["_successful"])
    values["family"] = values["problem"].map(
        {name: problem.family or name for name, problem in MULTI_OBJECTIVE_PROBLEMS.items()}
    )
    values["n_objectives"] = values["problem"].map(
        {name: problem.n_objectives for name, problem in MULTI_OBJECTIVE_PROBLEMS.items()}
    )
    return (
        values.groupby(["problem", "family", "n_objectives", "optimizer", "budget"])
        .agg(
            hv_gap_median=("normalized_hypervolume_gap", "median"),
            hv_gap_iqr=(
                "normalized_hypervolume_gap",
                lambda values: values.quantile(0.75) - values.quantile(0.25),
            ),
            igd_median=("normalized_igd", "median"),
            igd_iqr=(
                "normalized_igd",
                lambda values: values.quantile(0.75) - values.quantile(0.25),
            ),
            spacing_median=("spacing", "median"),
            spacing_valid_runs=("spacing", "count"),
            n_total_runs=("status", "size"),
            n_successful_runs=("_successful", "sum"),
            n_failed_runs=("_failed", "sum"),
        )
        .reset_index()
        .assign(success_rate=lambda summary: summary["n_successful_runs"] / summary["n_total_runs"])
    )


def write_metrics_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Write median and uncertainty summaries for each problem and budget."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_multiobjective_metrics(df)
    summary.to_csv(output_dir / "multi_objective_summary.csv", index=False)
    print(f"Saved MO summary table to {output_dir / 'multi_objective_summary.csv'}")
    return summary


def write_failure_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Persist every failed outcome so polished plots cannot hide failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [column for column in FAILURE_COLUMNS if column in df]
    failures = df.loc[df["status"].eq("error"), columns]
    failures.to_csv(output_dir / "multi_objective_failures.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-objective plots")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/plots"))
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    df = store.load_complete_multi()

    write_failure_table(df, args.output_dir)
    if df["problem"].isin(REPRESENTATIVE_FRONT_PROBLEMS).any():
        selections, front_points = representative_terminal_front_tables(df)
        selection_path = args.output_dir / "multi_objective_representative_front_selection.csv"
        selections.to_csv(selection_path, index=False)
        plot_representative_terminal_fronts(selections, front_points, args.output_dir)
        print(f"Saved representative-front selection audit to {selection_path}")
    for metric, label in PRIMARY_METRICS.items():
        plot_metric_by_budget(df, args.output_dir, metric, label)
    summary = write_metrics_table(df, args.output_dir)
    summary_metrics = {
        "hv_gap_median": ("normalized_hypervolume_gap", "normalized hypervolume gap"),
        "igd_median": ("normalized_igd", "normalized IGD"),
    }
    for summary_metric, (output_name, label) in summary_metrics.items():
        ranks = aggregate_family_balanced_metric_ranks(summary, summary_metric)
        ranks.to_csv(
            args.output_dir / f"multi_objective_family_balanced_{output_name}_ranks.csv",
            index=False,
        )
        plot_family_balanced_metric_ranks(ranks, args.output_dir, output_name, label)


if __name__ == "__main__":
    main()
