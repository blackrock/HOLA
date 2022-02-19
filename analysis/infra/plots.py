# Copyright 2021 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import numpy as np
import pandera.typing as pat
import plotly.graph_objects as go
from pandas import DataFrame

from analysis.infra.data_persister import Analysis2

DEFAULT_DIRECTORY = Path(__file__).parents[1] / "plots"


RENAMING = {
    "GA @ 3": "GeneticAlgo",
    "NelderMead @ 0": "NelderMead",
    "hooke @ 0": "HookeJeeves",
    "random_search @ 0": "RandSearch",
    "random_search @ 1": "RandSearchX2",
    "sobol @ 0": "Sobol",
    "hola @ 49": "HOLA",
    "tpe @ 0": "TPE",
    "PSO @ 0": "PSO",
    "IGR @ 0": "IGR",
}


def prepare_data(df: pat.DataFrame[Analysis2]) -> DataFrame:
    # RandomSearch Config 1 is double budget -> on the plot make it the same
    double_budget = (df["optimizer"] == "random_search") & (df["configuration"] == 1)
    df.loc[double_budget, "num_iterations"] /= 2

    df["optimizer_cfg"] = df["optimizer"] + " @ " + df["configuration"].astype(str)
    df = df.replace(RENAMING).sort_values(["benchmark", "num_iterations", "optimizer_cfg"])
    df["num_iterations"] = df["num_iterations"].astype(int).astype(str)  # Make num iterations appear as categorical
    return df


def generate_box_plots_by_num_iterations_per_benchmark(
    df: pat.DataFrame[Analysis2], directory: Path = DEFAULT_DIRECTORY
) -> None:
    """One file per function/benchmark which contains all numbers of iterations."""
    df = prepare_data(df)
    for key, dd in df.groupby("benchmark"):
        # fig = px.box(dd, x="num_iterations", y="best", color="optimizer_cfg", title=f'Benchmark: {key}')
        # fig.show()
        fig = go.Figure()
        fig.update_layout(title=f"Benchmark: {key}")
        n_colors = len(df["optimizer_cfg"].drop_duplicates())
        colors = [f"hsl({h},50%,60%)" for h in np.linspace(0, 360, n_colors)]
        for (optimizer_cfg, dd2), color in zip(dd.groupby("optimizer_cfg"), colors):
            fig.add_trace(
                go.Box(x=dd2["num_iterations"], y=dd2["best"], boxmean=True, name=optimizer_cfg, marker_color=color)
            )
        fig.update_layout(
            boxmode="group",
            yaxis_title="Best value found",
            xaxis_title="Number of iterations",
            legend_title_text="Optimizer",
        )
        fig.write_html(directory / f"{key}.html")
        fig.update_layout(width=1200, height=500, autosize=False)
        fig.write_image(directory / f"{key}.jpg", scale=10)


def generate_box_plots_per_benchmark_and_num_iterations(
    df: pat.DataFrame[Analysis2], directory: Path = DEFAULT_DIRECTORY
) -> None:
    """One file per function/benchmark and number of iterations."""
    df = prepare_data(df)
    # Generate plots
    for key, dd in df.groupby("benchmark"):
        for num_iterations, dd2 in dd.groupby("num_iterations"):
            fig = go.Figure()
            fig.update_layout(title=f"Benchmark: {key}, number of iterations: {num_iterations}")
            for optimizer_cfg, dd3 in dd2.groupby("optimizer_cfg"):
                fig.add_trace(go.Box(y=dd3["best"], boxmean=True, name=optimizer_cfg))
            fig.update_layout(yaxis_title="Best value found", legend_title_text="Optimizer")
            fig.write_html(directory / f"{key}-{num_iterations}.html")
