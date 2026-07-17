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
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.data.composite import load_reporting_results
from benchmarks.plotting.bootstrap import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    paired_median_summary,
    problem_context_seed,
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
HPO_GMM_OPTIMIZER = "HOLA HPO (GMM)"
HPO_GMM_COMPARATORS = (
    "HOLA HPO (random)",
    "HOLA HPO (sobol)",
    "Optuna HPO (TPE)",
)
HPO_PAIRED_DIFFERENCE_BOOTSTRAP_METHOD = (
    "paired run-id resampling; median of paired differences; percentile interval"
)
HPO_BUDGET_CHANGE = (25, 100)
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


def _validated_hpo_metric_values(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {"problem", "optimizer", "budget", "run_id", "status", metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"HPO paired contrast is missing columns: {', '.join(sorted(missing))}")
    if results.empty:
        raise ValueError("HPO paired contrast requires at least one result row")
    key_columns = ["problem", "optimizer", "budget", "run_id"]
    if results.duplicated(key_columns).any():
        raise ValueError("HPO paired contrast requires unique campaign run keys")
    invalid_status = results["status"].isna() | ~results["status"].isin({"success", "error"})
    if invalid_status.any():
        raise ValueError("HPO paired contrast encountered an invalid result status")

    values = results.copy()
    numeric_metric = pd.to_numeric(values[metric], errors="coerce")
    successful = values["status"].eq("success")
    if (successful & ~np.isfinite(numeric_metric)).any():
        raise ValueError(f"successful HPO runs must have a finite {metric}")
    values["_metric"] = numeric_metric.where(successful)

    for problem_key, problem_rows in values.groupby("problem", sort=True):
        problem = str(problem_key)
        expected_run_ids = set(int(value) for value in problem_rows["run_id"])
        for _, cell in problem_rows.groupby(["optimizer", "budget"], sort=False):
            if set(int(value) for value in cell["run_id"]) != expected_run_ids:
                raise ValueError(
                    f"problem {problem!r} does not have the same run IDs in every HPO cell"
                )
    return values


def _paired_hpo_cells(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    require_shared_search_seed: bool,
) -> pd.DataFrame:
    metadata = [
        column for column in ("search_seed", "split_seed") if column in left and column in right
    ]
    selected_columns = ["problem", "run_id", "status", "_metric", *metadata]
    paired = left[selected_columns].merge(
        right[selected_columns],
        on=["problem", "run_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_focal", "_reference"),
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("HPO paired contrast encountered unpaired runs")
    paired = paired.drop(columns="_merge")
    if (
        "split_seed_focal" in paired
        and not paired["split_seed_focal"].eq(paired["split_seed_reference"]).all()
    ):
        raise ValueError("HPO paired contrast requires a shared split seed")
    if (
        require_shared_search_seed
        and "search_seed_focal" in paired
        and not paired["search_seed_focal"].eq(paired["search_seed_reference"]).all()
    ):
        raise ValueError("HPO optimizer contrast requires a shared search seed")

    complete = paired["status_focal"].eq("success") & paired["status_reference"].eq("success")
    paired["_difference"] = (paired["_metric_focal"] - paired["_metric_reference"]).where(complete)
    return paired.sort_values("run_id", ignore_index=True)


def _paired_difference_statistics(
    paired: pd.DataFrame,
    *,
    draws: np.ndarray,
    confidence_level: float,
) -> dict[str, object]:
    differences = paired["_difference"].to_numpy(dtype=float)
    finite = np.isfinite(differences)
    if finite.any():
        estimate = float(np.nanmedian(differences))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bootstrap = np.nanmedian(differences[draws], axis=1)
        finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
        valid_resamples = len(finite_bootstrap)
        if valid_resamples:
            alpha = (1.0 - confidence_level) / 2.0
            lower, upper = np.quantile(finite_bootstrap, [alpha, 1.0 - alpha])
            ci_lower = float(lower)
            ci_upper = float(upper)
        else:
            ci_lower = float("nan")
            ci_upper = float("nan")
    else:
        estimate = float("nan")
        ci_lower = float("nan")
        ci_upper = float("nan")
        valid_resamples = 0

    return {
        "median_paired_difference": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_total_pairs": len(paired),
        "n_complete_pairs": int(finite.sum()),
        "n_incomplete_pairs": int((~finite).sum()),
        "n_focal_failures": int(paired["status_focal"].eq("error").sum()),
        "n_reference_failures": int(paired["status_reference"].eq("error").sum()),
        "bootstrap_valid_resamples": valid_resamples,
    }


def _bootstrap_draws(
    run_ids: Sequence[int],
    *,
    context_seed: int,
    n_resamples: int,
) -> np.ndarray:
    generator = np.random.default_rng(context_seed)
    return generator.integers(0, len(run_ids), size=(n_resamples, len(run_ids)))


def summarize_hpo_optimizer_contrasts(
    results: pd.DataFrame,
    *,
    focal_optimizer: str = HPO_GMM_OPTIMIZER,
    comparators: Sequence[str] = HPO_GMM_COMPARATORS,
    metrics: Sequence[str] = tuple(HPO_METRICS),
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Report paired GMM-minus-comparator median differences at each budget."""
    if (
        not comparators
        or focal_optimizer in comparators
        or len(set(comparators)) != len(comparators)
    ):
        raise ValueError("HPO optimizer contrast names must be non-empty and distinct")
    if not metrics:
        raise ValueError("HPO optimizer contrasts require at least one metric")
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between zero and one")

    rows: list[dict[str, object]] = []
    for metric in metrics:
        values = _validated_hpo_metric_values(results, metric)
        available = set(values["optimizer"])
        missing = {focal_optimizer, *comparators} - available
        if missing:
            raise ValueError("HPO optimizer contrasts omit " + ", ".join(sorted(missing)))
        for problem_key, problem_rows in values.groupby("problem", sort=True):
            problem = str(problem_key)
            run_ids = sorted(int(value) for value in problem_rows["run_id"].unique())
            context_seed = problem_context_seed(seed, problem)
            draws = _bootstrap_draws(
                run_ids,
                context_seed=context_seed,
                n_resamples=n_resamples,
            )
            for budget_key in sorted(problem_rows["budget"].unique()):
                budget = int(budget_key)
                focal = problem_rows[
                    problem_rows["optimizer"].eq(focal_optimizer)
                    & problem_rows["budget"].eq(budget)
                ]
                for comparator in comparators:
                    reference = problem_rows[
                        problem_rows["optimizer"].eq(comparator) & problem_rows["budget"].eq(budget)
                    ]
                    paired = _paired_hpo_cells(
                        focal,
                        reference,
                        require_shared_search_seed=True,
                    )
                    rows.append(
                        {
                            "problem": problem,
                            "metric": metric,
                            "metric_direction": "higher_is_better",
                            "contrast_type": "optimizer_at_fixed_budget",
                            "focal_optimizer": focal_optimizer,
                            "reference_optimizer": comparator,
                            "budget": budget,
                            "difference_definition": "focal_optimizer - reference_optimizer",
                            "positive_difference_favors": "focal_optimizer",
                            **_paired_difference_statistics(
                                paired,
                                draws=draws,
                                confidence_level=confidence_level,
                            ),
                            "confidence_level": confidence_level,
                            "bootstrap_resamples": n_resamples,
                            "bootstrap_method": HPO_PAIRED_DIFFERENCE_BOOTSTRAP_METHOD,
                            "bootstrap_seed": seed,
                            "bootstrap_context_seed": context_seed,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["metric", "reference_optimizer", "budget"],
        ignore_index=True,
    )


def summarize_hpo_budget_changes(
    results: pd.DataFrame,
    *,
    earlier_budget: int = HPO_BUDGET_CHANGE[0],
    later_budget: int = HPO_BUDGET_CHANGE[1],
    metrics: Sequence[str] = tuple(HPO_METRICS),
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Report paired later-minus-earlier HPO budget changes for every optimizer."""
    if later_budget <= earlier_budget:
        raise ValueError("the later HPO budget must exceed the earlier budget")
    if not metrics:
        raise ValueError("HPO budget changes require at least one metric")
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between zero and one")

    rows: list[dict[str, object]] = []
    for metric in metrics:
        values = _validated_hpo_metric_values(results, metric)
        missing_budgets = {earlier_budget, later_budget} - set(values["budget"])
        if missing_budgets:
            raise ValueError(
                "HPO budget changes omit budget(s) "
                + ", ".join(str(value) for value in sorted(missing_budgets))
            )
        for problem_key, problem_rows in values.groupby("problem", sort=True):
            problem = str(problem_key)
            run_ids = sorted(int(value) for value in problem_rows["run_id"].unique())
            context_seed = problem_context_seed(seed, problem)
            draws = _bootstrap_draws(
                run_ids,
                context_seed=context_seed,
                n_resamples=n_resamples,
            )
            for optimizer_key in sorted(problem_rows["optimizer"].unique()):
                optimizer = str(optimizer_key)
                later = problem_rows[
                    problem_rows["optimizer"].eq(optimizer)
                    & problem_rows["budget"].eq(later_budget)
                ]
                earlier = problem_rows[
                    problem_rows["optimizer"].eq(optimizer)
                    & problem_rows["budget"].eq(earlier_budget)
                ]
                paired = _paired_hpo_cells(
                    later,
                    earlier,
                    require_shared_search_seed=False,
                )
                statistics = _paired_difference_statistics(
                    paired,
                    draws=draws,
                    confidence_level=confidence_level,
                )
                rows.append(
                    {
                        "problem": problem,
                        "metric": metric,
                        "metric_direction": "higher_is_better",
                        "contrast_type": "paired_budget_change",
                        "optimizer": optimizer,
                        "earlier_budget": earlier_budget,
                        "later_budget": later_budget,
                        "difference_definition": "later_budget - earlier_budget",
                        "positive_difference_favors": "later_budget",
                        "median_paired_difference": statistics["median_paired_difference"],
                        "ci_lower": statistics["ci_lower"],
                        "ci_upper": statistics["ci_upper"],
                        "n_total_pairs": statistics["n_total_pairs"],
                        "n_complete_pairs": statistics["n_complete_pairs"],
                        "n_incomplete_pairs": statistics["n_incomplete_pairs"],
                        "n_later_budget_failures": statistics["n_focal_failures"],
                        "n_earlier_budget_failures": statistics["n_reference_failures"],
                        "bootstrap_valid_resamples": statistics["bootstrap_valid_resamples"],
                        "confidence_level": confidence_level,
                        "bootstrap_resamples": n_resamples,
                        "bootstrap_method": HPO_PAIRED_DIFFERENCE_BOOTSTRAP_METHOD,
                        "bootstrap_seed": seed,
                        "bootstrap_context_seed": context_seed,
                    }
                )
    return pd.DataFrame(rows).sort_values(["metric", "optimizer"], ignore_index=True)


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
    mechanism_optimizers = {HPO_GMM_OPTIMIZER, *HPO_GMM_COMPARATORS}
    if mechanism_optimizers.issubset(set(results["optimizer"])):
        optimizer_contrasts = summarize_hpo_optimizer_contrasts(
            results,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        optimizer_contrasts.to_csv(
            args.output_dir / "hpo_paired_optimizer_contrasts.csv",
            index=False,
        )
    else:
        missing = sorted(mechanism_optimizers - set(results["optimizer"]))
        print("Skipped HOLA HPO GMM contrasts; campaign omits " + ", ".join(missing))
    if set(HPO_BUDGET_CHANGE).issubset(set(int(value) for value in results["budget"])):
        budget_changes = summarize_hpo_budget_changes(
            results,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        budget_changes.to_csv(
            args.output_dir / "hpo_paired_budget_changes.csv",
            index=False,
        )
    else:
        print("Skipped HPO 25-to-100 budget contrasts; campaign omits budget 25 or 100")
    for metric in HPO_METRICS:
        plot_hpo_metric(summaries[metric], args.output_dir, metric)


if __name__ == "__main__":
    main()
