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

"""Single-objective plotting: box plots, normalized score plots, convergence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.data.normalize import aggregate_scores, min_max_normalize
from benchmarks.data.persistence import ResultStore
from benchmarks.plotting.style import apply_paper_style, get_color


def plot_normalized_means(df: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    """Bar plot of mean normalized score per optimizer, grouped by budget."""
    apply_paper_style()
    normed = min_max_normalize(df)
    scores = aggregate_scores(normed)

    budgets = sorted(scores["budget"].unique())
    optimizers = scores.groupby("optimizer")["mean_score"].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(6.3, 4), layout="constrained")
    width = 0.8 / len(optimizers)

    for i, opt in enumerate(optimizers):
        opt_data = scores[scores["optimizer"] == opt]
        positions = [budgets.index(b) + i * width for b in opt_data["budget"]]
        ax.bar(
            positions,
            opt_data["mean_score"],
            width=width,
            label=opt,
            color=get_color(opt),
        )

    ax.set_xticks([i + width * len(optimizers) / 2 for i in range(len(budgets))])
    ax.set_xticklabels(budgets)
    ax.set_xlabel("Iteration budget")
    ax.set_ylabel("Mean normalized score (lower is better)")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        borderaxespad=0,
    )
    ax.set_title("Average performance across benchmarks")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"mean_normalized{suffix}.pdf")
    fig.savefig(output_dir / f"mean_normalized{suffix}.png")
    fig.savefig(output_dir / f"mean_normalized{suffix}.pgf")
    plt.close(fig)
    print(f"Saved normalized mean plots to {output_dir}")


def plot_box_per_benchmark(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plot per benchmark showing distribution across runs."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    for problem_name, group in df.groupby("problem"):
        optimizers = sorted(group["optimizer"].unique())
        fig, ax = plt.subplots(figsize=(6.3, 4), layout="constrained")

        data_by_opt = [group[group["optimizer"] == opt]["best_value"].values for opt in optimizers]
        bp = ax.boxplot(
            data_by_opt,
            tick_labels=optimizers,
            showmeans=True,
            meanline=True,
            patch_artist=True,
        )
        for patch, opt in zip(bp["boxes"], optimizers, strict=True):
            patch.set_facecolor(get_color(opt))
            patch.set_alpha(0.7)

        ax.set_ylabel("Best value found (lower is better)")
        ax.set_title(f"Benchmark: {problem_name.replace('_', ' ')}")
        plt.xticks(rotation=45, ha="right")

        fig.savefig(output_dir / f"box_{problem_name}.pdf")
        fig.savefig(output_dir / f"box_{problem_name}.pgf")
        plt.close(fig)

    print(f"Saved per-benchmark box plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single-objective plots")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/plots"))
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    df = store.load_single()
    if df.empty:
        print("No single-objective results found.")
        return

    # Split into small and large budgets for the paper
    small = df[df["budget"] <= 75]
    large = df[df["budget"] >= 100]

    if not small.empty:
        plot_normalized_means(small, args.output_dir, suffix="_small_iter")
    if not large.empty:
        plot_normalized_means(large, args.output_dir, suffix="_large_iter")

    plot_box_per_benchmark(df, args.output_dir)


if __name__ == "__main__":
    main()
