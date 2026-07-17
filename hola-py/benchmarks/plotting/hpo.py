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

"""Authenticated reporting for the practical mixed-space HPO campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

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

plt.switch_backend("Agg")

HPO_METRICS = {
    "best_validation_r2": ("Validation $R^2$", "hpo_validation_r2_by_budget"),
    "heldout_test_r2": ("Held-out test $R^2$", "hpo_heldout_test_r2_by_budget"),
}
HPO_PROBLEM_LABELS = {
    "gbr_diabetes_hpo": "Gradient-boosted regression on diabetes",
}
HPO_FAILURE_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "run_id",
    "search_seed",
    "split_seed",
    "error",
    "n_validation_evaluations",
    "n_heldout_evaluations",
]


def summarize_hpo(
    results: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    """Keep validation-selection and held-out generalization summaries separate."""
    return {
        metric: paired_median_summary(
            results,
            metric,
            n_resamples=n_resamples,
            seed=seed,
        )
        for metric in HPO_METRICS
    }


def hpo_failure_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return one audit row for every failed HPO outcome."""
    columns = [column for column in HPO_FAILURE_COLUMNS if column in results]
    return results.loc[results["status"].eq("error"), columns].copy()


def plot_hpo_metric(summary: pd.DataFrame, output_dir: Path, metric: str) -> None:
    """Plot one HPO outcome without mixing validation and held-out scores."""
    label, filename = HPO_METRICS[metric]
    apply_paper_style()
    for problem_key, problem_rows in summary.groupby("problem", sort=True):
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

        ax.set_xlabel("Validation evaluations")
        ax.set_ylabel(label)
        ax.set_title(HPO_PROBLEM_LABELS.get(problem, problem.replace("_", " ")))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        save_figure(fig, output_dir, f"{filename}_{problem}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the authenticated practical HPO campaign")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark_results/hpo"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/hpo/plots"),
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
    results = load_reporting_results(args.results_dir, "hpo")
    summaries = summarize_hpo(
        results,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    failures = hpo_failure_table(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries["best_validation_r2"].to_csv(
        args.output_dir / "hpo_validation_summary.csv",
        index=False,
    )
    summaries["heldout_test_r2"].to_csv(
        args.output_dir / "hpo_heldout_summary.csv",
        index=False,
    )
    failures.to_csv(args.output_dir / "hpo_failures.csv", index=False)
    for metric in HPO_METRICS:
        plot_hpo_metric(summaries[metric], args.output_dir, metric)


if __name__ == "__main__":
    main()
