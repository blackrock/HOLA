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

"""Deterministic run-paired bootstrap summaries for secondary campaigns."""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_715
DEFAULT_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_METHOD = "paired run-id resampling; median; percentile interval"


def problem_context_seed(seed: int, problem: str) -> int:
    """Domain-separate a stable bootstrap stream for one problem."""
    material = f"paired-bootstrap:{seed}:{problem}".encode()
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def paired_median_summary(
    results: pd.DataFrame,
    metric: str,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Summarize medians with a deterministic bootstrap paired by ``run_id``.

    Within each problem, one shared matrix of resampled run IDs is used for
    every optimizer and budget. This preserves optimizer pairing and, for HPO,
    the fixed data-split pairing across budgets. Failed runs remain in the
    resampling frame as missing metric outcomes and are counted explicitly.
    """
    required = {"problem", "optimizer", "budget", "run_id", "status", metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"paired bootstrap is missing columns: {', '.join(sorted(missing))}")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples <= 0
    ):
        raise ValueError("bootstrap resamples must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between zero and one")
    if results.empty:
        raise ValueError("paired bootstrap requires at least one result row")

    key_columns = ["problem", "optimizer", "budget", "run_id"]
    if results.duplicated(key_columns).any():
        raise ValueError("paired bootstrap requires unique campaign run keys")
    invalid_status = ~results["status"].isin({"success", "error"})
    if invalid_status.any():
        raise ValueError("paired bootstrap encountered an invalid result status")

    values = results.copy()
    numeric_metric = pd.to_numeric(values[metric], errors="coerce")
    successful = values["status"].eq("success")
    invalid_success = successful & ~np.isfinite(numeric_metric)
    if invalid_success.any():
        raise ValueError(f"successful runs must have a finite {metric}")
    values["_metric"] = numeric_metric.where(successful)

    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, object]] = []
    for problem_key, problem_rows in values.groupby("problem", sort=True):
        problem = str(problem_key)
        run_ids = sorted(int(run_id) for run_id in problem_rows["run_id"].unique())
        if not run_ids:
            raise ValueError(f"problem {problem!r} has no run IDs")
        expected_run_ids = set(run_ids)
        for _, cell in problem_rows.groupby(["optimizer", "budget"], sort=False):
            if set(int(run_id) for run_id in cell["run_id"]) != expected_run_ids:
                raise ValueError(
                    f"problem {problem!r} does not have the same run IDs in every cell"
                )

        context_seed = problem_context_seed(seed, problem)
        generator = np.random.default_rng(context_seed)
        draws = generator.integers(0, len(run_ids), size=(n_resamples, len(run_ids)))

        for (optimizer_key, budget_key), cell in problem_rows.groupby(
            ["optimizer", "budget"],
            sort=True,
        ):
            optimizer = str(optimizer_key)
            budget = int(budget_key)
            aligned = cell.set_index("run_id").reindex(run_ids)
            metric_values = aligned["_metric"].to_numpy(dtype=float)
            n_successful = int(np.isfinite(metric_values).sum())
            n_total = len(metric_values)
            n_failed = n_total - n_successful
            if n_successful:
                estimate = float(np.nanmedian(metric_values))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bootstrap = np.nanmedian(metric_values[draws], axis=1)
                finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
                valid_resamples = len(finite_bootstrap)
                if valid_resamples:
                    lower, upper = np.quantile(
                        finite_bootstrap,
                        [alpha, 1.0 - alpha],
                    )
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

            rows.append(
                {
                    "problem": problem,
                    "optimizer": optimizer,
                    "budget": budget,
                    "metric": metric,
                    "median": estimate,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_total_runs": n_total,
                    "n_successful_runs": n_successful,
                    "n_failed_runs": n_failed,
                    "success_rate": n_successful / n_total,
                    "bootstrap_resamples": n_resamples,
                    "bootstrap_valid_resamples": valid_resamples,
                    "confidence_level": confidence_level,
                    "bootstrap_method": BOOTSTRAP_METHOD,
                    "bootstrap_seed": seed,
                    "bootstrap_context_seed": context_seed,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["problem", "budget", "optimizer"],
        ignore_index=True,
    )
