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

"""Single-objective plotting based on fixed known-optimum regret."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.data.normalize import (
    GMM_MECHANISM_COMPARATORS,
    GMM_OPTIMIZER,
    add_simple_regret,
    aggregate_family_balanced_paired_win_rates,
    aggregate_family_balanced_ranks,
    summarize_regret,
)
from benchmarks.data.persistence import ResultStore
from benchmarks.plotting.export import save_figure
from benchmarks.plotting.style import apply_paper_style, get_color
from benchmarks.problems.single_objective import SINGLE_OBJECTIVE_PROBLEMS

plt.switch_backend("Agg")

FAILURE_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "run_id",
    "seed",
    "error",
    "n_evaluations",
]


def plot_regret_by_family(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot median regret by budget without pooling families or dimensions."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, group in summary.groupby(["suite", "family", "dimension"]):
        suite, family, dimension = cast(tuple[str, str, int], key)
        budgets = sorted(group["budget"].unique())
        optimizers = sorted(group["optimizer"].unique())
        width = 0.8 / len(optimizers)
        fig, ax = plt.subplots(figsize=(6.3, 4), layout="constrained")

        for index, optimizer in enumerate(optimizers):
            optimizer_rows = group[group["optimizer"] == optimizer].set_index("budget")
            present_budgets = [budget for budget in budgets if budget in optimizer_rows.index]
            rows = optimizer_rows.loc[present_budgets]
            medians = rows["regret_median"].to_numpy(dtype=float)
            errors = np.vstack(
                [
                    medians - rows["regret_q1"].to_numpy(dtype=float),
                    rows["regret_q3"].to_numpy(dtype=float) - medians,
                ]
            )
            positions = [budgets.index(budget) + index * width for budget in present_budgets]
            bars = ax.bar(
                positions,
                medians,
                width=width,
                yerr=errors,
                capsize=2,
                label=optimizer,
                color=get_color(optimizer),
            )
            if optimizer == "Random x2":
                actual_evaluations = rows["actual_evaluations_median"].to_numpy(dtype=float)
                ax.bar_label(
                    bars,
                    labels=[f"{value:g} evals" for value in actual_evaluations],
                    fontsize=6,
                    rotation=90,
                    padding=2,
                )

        ax.set_xticks([index + width * len(optimizers) / 2 for index in range(len(budgets))])
        ax.set_xticklabels(budgets)
        ax.set_xlabel("Declared evaluation budget (Random x2 uses 2x)")
        ax.set_ylabel("Simple regret (lower is better)")
        ax.set_title(f"{family.replace('_', ' ')}, {dimension}D")
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            borderaxespad=0,
        )
        save_figure(fig, output_dir, f"regret_{suite}_{family}_{dimension}d")
        plt.close(fig)

    print(f"Saved family- and dimension-specific regret plots to {output_dir}")


def plot_box_per_benchmark(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot regret distributions separately for every problem and budget."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    finite = df[np.isfinite(pd.to_numeric(df["regret"], errors="coerce"))]

    for (problem_name, budget), group in finite.groupby(["problem", "budget"]):
        name = str(problem_name)
        optimizers = sorted(group["optimizer"].unique())
        fig, ax = plt.subplots(figsize=(6.3, 4), layout="constrained")

        optimizer_groups = [group[group["optimizer"] == optimizer] for optimizer in optimizers]
        data_by_optimizer = [
            optimizer_group["regret"].values for optimizer_group in optimizer_groups
        ]
        tick_labels = []
        for optimizer, optimizer_group in zip(optimizers, optimizer_groups, strict=True):
            actual = (
                pd.to_numeric(optimizer_group["n_evaluations"], errors="coerce").median()
                if "n_evaluations" in optimizer_group
                else float("nan")
            )
            tick_labels.append(
                f"{optimizer}\n({actual:g} evals)" if pd.notna(actual) else optimizer
            )
        boxes = ax.boxplot(
            data_by_optimizer,
            tick_labels=tick_labels,
            showmeans=True,
            meanline=True,
            patch_artist=True,
        )
        for patch, optimizer in zip(boxes["boxes"], optimizers, strict=True):
            patch.set_facecolor(get_color(optimizer))
            patch.set_alpha(0.7)

        ax.set_ylabel("Simple regret (lower is better)")
        ax.set_title(f"{name.replace('_', ' ')}, {budget} evaluations")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, output_dir, f"box_regret_{name}_{budget}eval")
        plt.close(fig)

    print(f"Saved per-problem, per-budget regret plots to {output_dir}")


def plot_family_balanced_ranks(ranks: pd.DataFrame, output_dir: Path) -> None:
    """Plot the compact family-balanced headline comparison by suite."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite_key, group in ranks.groupby("suite"):
        suite = str(suite_key)
        budgets = sorted(group["budget"].unique())
        optimizers = sorted(group["optimizer"].unique())
        width = 0.8 / len(optimizers)
        fig, ax = plt.subplots(figsize=(6.3, 4), layout="constrained")
        for index, optimizer in enumerate(optimizers):
            rows = group[group["optimizer"] == optimizer].set_index("budget")
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
        ax.set_title(f"{suite.capitalize()} benchmark summary")
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            borderaxespad=0,
        )
        save_figure(fig, output_dir, f"family_balanced_rank_{suite}")
        plt.close(fig)

    print(f"Saved family-balanced rank plots to {output_dir}")


def plot_gmm_paired_win_rates(win_rates: pd.DataFrame, output_dir: Path) -> None:
    """Plot the family-balanced paired GMM mechanism comparison."""
    required = {
        "budget",
        "comparator",
        "win_rate",
        "ci_lower",
        "ci_upper",
    }
    missing = required - set(win_rates.columns)
    if missing:
        raise ValueError(f"missing paired-win columns: {', '.join(sorted(missing))}")

    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 4), layout="constrained")
    for comparator in GMM_MECHANISM_COMPARATORS:
        rows = win_rates[win_rates["comparator"].eq(comparator)].sort_values("budget")
        if rows.empty:
            continue
        rates = rows["win_rate"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                rates - rows["ci_lower"].to_numpy(dtype=float),
                rows["ci_upper"].to_numpy(dtype=float) - rates,
            ]
        )
        ax.errorbar(
            rows["budget"].to_numpy(dtype=int),
            rates,
            yerr=errors,
            marker="o",
            capsize=3,
            label=f"versus {comparator}",
            color=get_color(comparator),
        )

    ax.axhline(0.5, color="#4d4d4d", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(sorted(win_rates["budget"].unique()))
    ax.set_xlabel("Declared evaluation budget")
    ax.set_ylabel("Family-balanced paired win rate")
    ax.set_title("HOLA GMM versus exploration baselines")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        borderaxespad=0,
    )
    save_figure(fig, output_dir, "gmm_family_balanced_paired_win_rate")
    plt.close(fig)
    print(f"Saved HOLA GMM paired-win plot to {output_dir}")


def write_failure_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Persist every failed outcome so polished plots cannot hide failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [column for column in FAILURE_COLUMNS if column in df]
    failures = df.loc[df["status"].eq("error"), columns]
    failures.to_csv(output_dir / "single_objective_failures.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single-objective plots")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/plots"))
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    df = store.load_complete_single()

    regret = add_simple_regret(df, SINGLE_OBJECTIVE_PROBLEMS)
    summary = summarize_regret(regret)
    ranks = aggregate_family_balanced_ranks(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_failure_table(df, args.output_dir)
    summary.to_csv(args.output_dir / "single_objective_regret_summary.csv", index=False)
    ranks.to_csv(args.output_dir / "single_objective_family_balanced_ranks.csv", index=False)
    mechanism_optimizers = {GMM_OPTIMIZER, *GMM_MECHANISM_COMPARATORS}
    if mechanism_optimizers.issubset(set(regret["optimizer"])):
        win_rates = aggregate_family_balanced_paired_win_rates(regret)
        win_rates.to_csv(
            args.output_dir / "single_objective_gmm_paired_win_rates.csv",
            index=False,
        )
        plot_gmm_paired_win_rates(win_rates, args.output_dir)
    else:
        missing = sorted(mechanism_optimizers - set(regret["optimizer"]))
        print("Skipped HOLA GMM mechanism comparison; campaign omits " + ", ".join(missing))
    plot_family_balanced_ranks(ranks, args.output_dir)
    plot_regret_by_family(summary, args.output_dir)
    plot_box_per_benchmark(regret, args.output_dir)


if __name__ == "__main__":
    main()
