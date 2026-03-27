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

"""Multi-objective plotting: HV/IGD bar plots, Pareto front overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.data.persistence import ResultStore
from benchmarks.plotting.style import apply_paper_style, get_color


def plot_hv_by_budget(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of median hypervolume per optimizer, grouped by budget."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    for problem_name, group in df.groupby("problem"):
        fig, ax = plt.subplots(figsize=(6.3, 3.5), layout="constrained")
        budgets = sorted(group["budget"].unique())
        optimizers = sorted(group["optimizer"].unique())
        width = 0.8 / len(optimizers)

        for i, opt in enumerate(optimizers):
            opt_data = group[group["optimizer"] == opt]
            medians = [opt_data[opt_data["budget"] == b]["hypervolume"].median() for b in budgets]
            positions = [j + i * width for j in range(len(budgets))]
            ax.bar(positions, medians, width=width, label=opt, color=get_color(opt))

        ax.set_xticks([j + width * len(optimizers) / 2 for j in range(len(budgets))])
        ax.set_xticklabels(budgets)
        ax.set_xlabel("Iteration budget")
        ax.set_ylabel("Hypervolume (higher is better)")
        ax.set_title(f"{problem_name.replace('_', ' ')}")
        ax.legend(
            fontsize=7,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

        fig.savefig(output_dir / f"hv_{problem_name}.pdf")
        fig.savefig(output_dir / f"hv_{problem_name}.pgf")
        plt.close(fig)

    print(f"Saved HV plots to {output_dir}")


def plot_metrics_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary table of median metrics per (problem, optimizer, budget)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        df.groupby(["problem", "optimizer", "budget"])
        .agg(
            hv_median=("hypervolume", "median"),
            hv_iqr=("hypervolume", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            igd_median=("igd", "median"),
            spacing_median=("spacing", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "mo_summary.csv", index=False)
    print(f"Saved MO summary table to {output_dir / 'mo_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-objective plots")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/plots"))
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    df = store.load_multi()
    if df.empty:
        print("No multi-objective results found.")
        return

    plot_hv_by_budget(df, args.output_dir)
    plot_metrics_table(df, args.output_dir)


if __name__ == "__main__":
    main()
