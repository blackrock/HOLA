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

"""Authenticated reporting for the explicitly grouped-TLP capability study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.data.composite import load_reporting_results
from benchmarks.plotting.bootstrap import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    paired_median_summary,
)
from benchmarks.plotting.export import save_figure
from benchmarks.plotting.style import apply_paper_style, get_color
from benchmarks.problems.grouped_tlp import GROUPED_TLP_PROBLEMS

plt.switch_backend("Agg")

GROUPED_METRICS = {
    "normalized_hypervolume_gap": "Normalized hypervolume gap",
    "normalized_igd": "Normalized IGD",
}
REPRESENTATIVE_SELECTION_RULE = (
    "minimum absolute distance to the method's median normalized hypervolume gap "
    "at the largest budget; ties by lower run_id, then lower seed"
)
GROUPED_FAILURE_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "run_id",
    "seed",
    "error",
    "n_evaluations",
]


def _validate_grouped_problems(results: pd.DataFrame) -> None:
    unknown = sorted(set(results["problem"].dropna()) - set(GROUPED_TLP_PROBLEMS))
    if unknown:
        raise ValueError(
            "grouped-TLP reporting received non-grouped problem(s): " + ", ".join(unknown)
        )


def summarize_grouped_tlp(
    results: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Return a failure-visible, long-form HV-gap and IGD summary."""
    _validate_grouped_problems(results)
    summaries = [
        paired_median_summary(
            results,
            metric,
            n_resamples=n_resamples,
            seed=seed,
        )
        for metric in GROUPED_METRICS
    ]
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["problem", "metric", "budget", "optimizer"],
        ignore_index=True,
    )


def grouped_failure_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return one audit row for every failed grouped-TLP outcome."""
    columns = [column for column in GROUPED_FAILURE_COLUMNS if column in results]
    return results.loc[results["status"].eq("error"), columns].copy()


def _parse_front(value: Any, n_objectives: int) -> np.ndarray:
    parsed = json.loads(value) if isinstance(value, str) else value
    front = np.asarray(parsed, dtype=float)
    if front.ndim != 2 or front.shape[1] != n_objectives or not np.all(np.isfinite(front)):
        raise ValueError(f"representative Pareto front must have shape (n, {n_objectives})")
    if len(front) == 0:
        raise ValueError("representative Pareto front must be non-empty")
    return front


def representative_front_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select and fully record one median-HV-gap run per method.

    Selection is performed independently for every problem at its largest
    reported budget according to :data:`REPRESENTATIVE_SELECTION_RULE`.
    """
    _validate_grouped_problems(results)
    selection_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    for problem_key, problem_rows in results.groupby("problem", sort=True):
        problem_name = str(problem_key)
        problem = GROUPED_TLP_PROBLEMS[problem_name]
        if problem.n_groups != 2:
            raise ValueError("representative grouped-TLP front plots require exactly two groups")
        group_1, group_2 = problem.group_names
        largest_budget = int(problem_rows["budget"].max())
        largest = problem_rows[problem_rows["budget"].eq(largest_budget)]

        for optimizer_key, optimizer_rows in largest.groupby("optimizer", sort=True):
            optimizer = str(optimizer_key)
            metric = pd.to_numeric(
                optimizer_rows["normalized_hypervolume_gap"],
                errors="coerce",
            )
            candidates = optimizer_rows[
                optimizer_rows["status"].eq("success") & np.isfinite(metric)
            ].copy()
            if candidates.empty:
                selection_rows.append(
                    {
                        "problem": problem_name,
                        "optimizer": optimizer,
                        "budget": largest_budget,
                        "selection_status": "no_successful_finite_run",
                        "run_id": None,
                        "seed": None,
                        "normalized_hypervolume_gap": None,
                        "method_median_hypervolume_gap": None,
                        "absolute_distance_to_median": None,
                        "n_pareto_points": None,
                        "selection_rule": REPRESENTATIVE_SELECTION_RULE,
                    }
                )
                continue

            candidates["normalized_hypervolume_gap"] = pd.to_numeric(
                candidates["normalized_hypervolume_gap"]
            )
            median_gap = float(candidates["normalized_hypervolume_gap"].median())
            candidates["_distance"] = (candidates["normalized_hypervolume_gap"] - median_gap).abs()
            minimum_distance = float(candidates["_distance"].min())
            tied = candidates[
                np.isclose(
                    candidates["_distance"],
                    minimum_distance,
                    rtol=0.0,
                    atol=1e-15,
                )
            ]
            chosen = tied.sort_values(
                ["run_id", "seed"],
                kind="mergesort",
            ).iloc[0]
            front = _parse_front(chosen["pareto_front"], problem.n_groups)
            reported_n_points = chosen.get("n_pareto_points")
            if pd.notna(reported_n_points) and int(reported_n_points) != len(front):
                raise ValueError(
                    "representative Pareto front point count does not match "
                    "the stored n_pareto_points"
                )
            run_id = int(chosen["run_id"])
            run_seed = int(chosen["seed"])
            selection_rows.append(
                {
                    "problem": problem_name,
                    "optimizer": optimizer,
                    "budget": largest_budget,
                    "selection_status": "selected",
                    "run_id": run_id,
                    "seed": run_seed,
                    "normalized_hypervolume_gap": float(chosen["normalized_hypervolume_gap"]),
                    "method_median_hypervolume_gap": median_gap,
                    "absolute_distance_to_median": float(chosen["_distance"]),
                    "n_pareto_points": len(front),
                    "selection_rule": REPRESENTATIVE_SELECTION_RULE,
                }
            )
            for point_index, point in enumerate(front):
                point_rows.append(
                    {
                        "problem": problem_name,
                        "budget": largest_budget,
                        "source": "selected_method_run",
                        "optimizer": optimizer,
                        "run_id": run_id,
                        "seed": run_seed,
                        "point_index": point_index,
                        "group_1_name": group_1,
                        "group_2_name": group_2,
                        "group_1_cost": float(point[0]),
                        "group_2_cost": float(point[1]),
                    }
                )

        for point_index, point in enumerate(problem.true_pareto_front):
            point_rows.append(
                {
                    "problem": problem_name,
                    "budget": largest_budget,
                    "source": "analytic_pareto_front",
                    "optimizer": "Analytic Pareto front",
                    "run_id": None,
                    "seed": None,
                    "point_index": point_index,
                    "group_1_name": group_1,
                    "group_2_name": group_2,
                    "group_1_cost": float(point[0]),
                    "group_2_cost": float(point[1]),
                }
            )

    return pd.DataFrame(selection_rows), pd.DataFrame(point_rows)


def plot_grouped_metric(summary: pd.DataFrame, output_dir: Path, metric: str) -> None:
    """Plot one lower-is-better grouped-TLP metric with paired intervals."""
    label = GROUPED_METRICS[metric]
    apply_paper_style()
    metric_rows = summary[summary["metric"].eq(metric)]
    for problem_key, problem_rows in metric_rows.groupby("problem", sort=True):
        problem = str(problem_key)
        fig, ax = plt.subplots(figsize=(6.3, 3.7), layout="constrained")
        for optimizer_key, optimizer_rows in problem_rows.groupby("optimizer", sort=True):
            optimizer = str(optimizer_key)
            ordered = optimizer_rows.sort_values("budget")
            finite = np.isfinite(ordered["median"].to_numpy(dtype=float))
            plotted = ordered.loc[finite]
            if plotted.empty:
                continue
            medians = plotted["median"].to_numpy(dtype=float)
            errors = np.vstack(
                [
                    medians - plotted["ci_lower"].to_numpy(dtype=float),
                    plotted["ci_upper"].to_numpy(dtype=float) - medians,
                ]
            )
            ax.errorbar(
                plotted["budget"].to_numpy(dtype=int),
                medians,
                yerr=errors,
                marker="o",
                capsize=3,
                label=optimizer,
                color=get_color(optimizer),
            )
        ax.set_xlabel("Completed objective evaluations")
        ax.set_ylabel(f"{label} (lower is better)")
        ax.set_title(problem.replace("_", " "))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        save_figure(fig, output_dir, f"grouped_tlp_{metric}_{problem}")
        plt.close(fig)


def plot_representative_fronts(points: pd.DataFrame, output_dir: Path) -> None:
    """Plot audited selected runs over the analytic group-cost Pareto curve."""
    apply_paper_style()
    for key, rows in points.groupby(["problem", "budget"], sort=True):
        problem, budget = cast(tuple[str, int], key)
        group_1 = str(rows["group_1_name"].iloc[0])
        group_2 = str(rows["group_2_name"].iloc[0])
        fig, ax = plt.subplots(figsize=(5.4, 4.3), layout="constrained")
        analytic = rows[rows["source"].eq("analytic_pareto_front")].sort_values("point_index")
        ax.plot(
            analytic["group_1_cost"],
            analytic["group_2_cost"],
            color="black",
            linestyle="--",
            label="Analytic Pareto front",
            zorder=1,
        )
        selected = rows[rows["source"].eq("selected_method_run")]
        for optimizer_key, optimizer_rows in selected.groupby("optimizer", sort=True):
            optimizer = str(optimizer_key)
            ax.scatter(
                optimizer_rows["group_1_cost"],
                optimizer_rows["group_2_cost"],
                s=18,
                alpha=0.75,
                label=optimizer,
                color=get_color(optimizer),
                zorder=2,
            )
        ax.set_xlabel(f"{group_1.replace('_', ' ').title()} cost")
        ax.set_ylabel(f"{group_2.replace('_', ' ').title()} cost")
        ax.set_title(f"Representative fronts, {budget} evaluations")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        save_figure(
            fig,
            output_dir,
            f"grouped_tlp_representative_front_{problem}_{budget}eval",
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the authenticated grouped-TLP campaign")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark_results/grouped_tlp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/grouped_tlp/plots"),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    # Authenticate the manifest and exact Cartesian result coverage before
    # creating any reporting artifacts.
    results = load_reporting_results(args.results_dir, "multi_objective")
    summary = summarize_grouped_tlp(
        results,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    failures = grouped_failure_table(results)
    selections, front_points = representative_front_tables(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "grouped_tlp_summary.csv", index=False)
    failures.to_csv(args.output_dir / "grouped_tlp_failures.csv", index=False)
    selections.to_csv(
        args.output_dir / "grouped_tlp_representative_front_selection.csv",
        index=False,
    )
    front_points.to_csv(
        args.output_dir / "grouped_tlp_representative_front_points.csv",
        index=False,
    )
    for metric in GROUPED_METRICS:
        plot_grouped_metric(summary, args.output_dir, metric)
    plot_representative_fronts(front_points, args.output_dir)


if __name__ == "__main__":
    main()
