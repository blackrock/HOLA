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

"""Matplotlib style configuration for LaTeX-quality figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Color palette for optimizers (colorblind-friendly)
OPTIMIZER_COLORS: dict[str, str] = {
    "HOLA (auto)": "#1f77b4",
    "HOLA (sobol)": "#aec7e8",
    "HOLA (random)": "#c7c7c7",
    "HOLA MO (auto)": "#1f77b4",
    "HOLA MO (sobol)": "#aec7e8",
    "HOLA MO (random)": "#c7c7c7",
    "Random x2": "#9467bd",
    "TPE": "#ff7f0e",
    "IGR": "#2ca02c",
    "GA": "#d62728",
    "PSO": "#e377c2",
    "Nelder-Mead": "#8c564b",
    "Hooke-Jeeves": "#bcbd22",
    "NSGA-II (Optuna)": "#ff7f0e",
    "NSGA-II (pymoo)": "#d62728",
    "MOEA/D": "#2ca02c",
}


def apply_paper_style() -> None:
    """Apply matplotlib rcParams for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "figure.figsize": (8, 4),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "grid.alpha": 0.3,
        }
    )


def get_color(optimizer_name: str) -> str:
    """Get color for an optimizer, with fallback."""
    return OPTIMIZER_COLORS.get(optimizer_name, "#7f7f7f")
