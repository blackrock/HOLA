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

"""Min-max normalization and aggregation for benchmark results."""

from __future__ import annotations

import pandas as pd


def min_max_normalize(
    df: pd.DataFrame,
    value_col: str = "best_value",
    group_cols: tuple[str, ...] = ("problem", "budget"),
) -> pd.DataFrame:
    """Per (problem, budget), normalize value_col to [0, 1] via min-max.

    0 = best, 1 = worst across all optimizers for that (problem, budget).
    """
    result = df.copy()
    group = result.groupby(list(group_cols))[value_col]
    min_val = group.transform("min")
    max_val = group.transform("max")
    denom = max_val - min_val
    # Avoid division by zero (all same value)
    result["normalized"] = (result[value_col] - min_val) / denom.replace(0, 1)
    return result


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Mean normalized score per (optimizer, budget), averaging across problems.

    This is the "mean of means" metric used in the paper.
    """
    # First: mean per (optimizer, budget, problem)
    per_problem = df.groupby(["optimizer", "budget", "problem"])["normalized"].mean().reset_index()
    # Then: mean across problems per (optimizer, budget)
    return (
        per_problem.groupby(["optimizer", "budget"])["normalized"]
        .mean()
        .reset_index()
        .rename(columns={"normalized": "mean_score"})
        .sort_values(["budget", "mean_score"])
    )
