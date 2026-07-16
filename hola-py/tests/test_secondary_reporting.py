# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Reporting tests for the HPO and grouped-TLP secondary campaigns."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import benchmarks.plotting.grouped_tlp as grouped_reporting
import benchmarks.plotting.hpo as hpo_reporting
from benchmarks.data.manifest import build_campaign_manifest
from benchmarks.data.persistence import ResultStore
from benchmarks.data.seeding import make_hpo_split_seed, make_seed
from benchmarks.plotting import bootstrap as bootstrap_reporting
from benchmarks.plotting.bootstrap import paired_median_summary
from benchmarks.problems.grouped_tlp import SYNTHETIC_GROUPED_TLP

pytestmark = pytest.mark.benchmarks

TEST_PROVENANCE = {
    "code": {"commit": "test", "dirty": False, "source_hash": "test"},
    "lock_hash": "test",
    "python": {"implementation": "CPython", "version": "test"},
    "platform": {"platform": "test", "machine": "test", "system": "test"},
    "dependencies": {"hola-opt": None},
    "native_extension": None,
}


def test_paired_bootstrap_is_deterministic_and_failure_visible() -> None:
    rows = []
    for optimizer, offset in (("left", 0.0), ("right", 1.0)):
        for budget in (25, 50):
            for run_id in range(4):
                failed = optimizer == "right" and budget == 50 and run_id == 3
                rows.append(
                    {
                        "problem": "problem",
                        "optimizer": optimizer,
                        "budget": budget,
                        "run_id": run_id,
                        "status": "error" if failed else "success",
                        "metric": None if failed else offset + budget / 100 + run_id,
                    }
                )
    results = pd.DataFrame(rows)
    first = paired_median_summary(results, "metric", n_resamples=256, seed=123)
    second = paired_median_summary(results, "metric", n_resamples=256, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert first["bootstrap_context_seed"].nunique() == 1
    failed_cell = first[(first["optimizer"] == "right") & (first["budget"] == 50)].iloc[0]
    assert failed_cell["n_total_runs"] == 4
    assert failed_cell["n_successful_runs"] == 3
    assert failed_cell["n_failed_runs"] == 1
    assert failed_cell["bootstrap_valid_resamples"] <= 256


def test_paired_bootstrap_records_an_all_failure_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailureOnlyGenerator:
        def integers(
            self,
            low: int,
            high: int,
            *,
            size: tuple[int, int],
        ) -> np.ndarray:
            del low, high
            return np.ones(size, dtype=int)

    monkeypatch.setattr(
        bootstrap_reporting.np.random,
        "default_rng",
        lambda seed: _FailureOnlyGenerator(),
    )
    results = pd.DataFrame(
        [
            {
                "problem": "problem",
                "optimizer": "optimizer",
                "budget": 10,
                "run_id": 0,
                "status": "success",
                "metric": 1.0,
            },
            {
                "problem": "problem",
                "optimizer": "optimizer",
                "budget": 10,
                "run_id": 1,
                "status": "error",
                "metric": None,
            },
        ]
    )

    summary = paired_median_summary(results, "metric", n_resamples=1)

    assert summary.loc[0, "median"] == 1.0
    assert summary.loc[0, "bootstrap_valid_resamples"] == 0
    assert pd.isna(summary.loc[0, "ci_lower"])
    assert pd.isna(summary.loc[0, "ci_upper"])


def _hpo_manifest() -> dict[str, Any]:
    optimizers = ["left", "right"]
    budgets = [25, 50]
    return build_campaign_manifest(
        run_kind="hpo",
        budgets=budgets,
        n_runs=3,
        problem_names=["gbr_diabetes_hpo"],
        optimizer_names=optimizers,
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [
                    {"budget": budget, "configuration": {"adapter": optimizer}}
                    for budget in budgets
                ],
            }
            for optimizer in optimizers
        ],
        provenance=TEST_PROVENANCE,
    )


def _write_hpo_campaign(path: Path) -> ResultStore:
    store = ResultStore(path)
    store.prepare_campaign(_hpo_manifest(), resume=False)
    for optimizer_index, optimizer in enumerate(("left", "right")):
        for budget in (25, 50):
            for run_id in range(3):
                failed = optimizer == "right" and budget == 25 and run_id == 2
                store.append_hpo(
                    {
                        "problem": "gbr_diabetes_hpo",
                        "optimizer": optimizer,
                        "budget": budget,
                        "run_id": run_id,
                        "search_seed": make_seed("gbr_diabetes_hpo", budget, run_id),
                        "split_seed": make_hpo_split_seed("gbr_diabetes_hpo", run_id),
                        "status": "error" if failed else "success",
                        "error": "deliberate" if failed else "",
                        "optimizer_config": {"adapter": optimizer},
                        "n_validation_evaluations": 10 if failed else budget,
                        "best_validation_r2": (
                            None if failed else 0.4 + 0.01 * run_id + 0.02 * optimizer_index
                        ),
                        "best_params": None if failed else {"n_estimators": 50},
                        "validation_trace": None if failed else [0.4] * budget,
                        "heldout_test_r2": (
                            None if failed else 0.3 + 0.01 * run_id + 0.02 * optimizer_index
                        ),
                        "n_heldout_evaluations": 0 if failed else 1,
                        "train_size": 264,
                        "validation_size": 89,
                        "test_size": 89,
                        "wall_time_seconds": 1.0,
                    }
                )
    return store


def test_hpo_cli_writes_separate_authenticated_audit_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "plots"
    _write_hpo_campaign(results_dir)
    plotted: list[str] = []
    monkeypatch.setattr(
        hpo_reporting,
        "plot_hpo_metric",
        lambda summary, output, metric: plotted.append(metric),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot-hpo",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--bootstrap-resamples",
            "128",
        ],
    )

    hpo_reporting.main()

    validation = pd.read_csv(output_dir / "hpo_validation_summary.csv")
    heldout = pd.read_csv(output_dir / "hpo_heldout_summary.csv")
    failures = pd.read_csv(output_dir / "hpo_failures.csv")
    assert set(validation["metric"]) == {"best_validation_r2"}
    assert set(heldout["metric"]) == {"heldout_test_r2"}
    assert set(validation["bootstrap_method"]) == {
        "paired run-id resampling; median; percentile interval"
    }
    assert failures[["optimizer", "budget", "run_id"]].to_records(index=False).tolist() == [
        ("right", 25, 2)
    ]
    assert failures.loc[0, "search_seed"] != failures.loc[0, "split_seed"]
    assert plotted == ["best_validation_r2", "heldout_test_r2"]


def test_hpo_cli_authenticates_completeness_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "partial"
    output_dir = tmp_path / "plots"
    store = ResultStore(results_dir)
    store.prepare_campaign(_hpo_manifest(), resume=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot-hpo", "--results-dir", str(results_dir), "--output-dir", str(output_dir)],
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        hpo_reporting.main()
    assert not output_dir.exists()


def _grouped_manifest() -> dict[str, Any]:
    optimizers = ["HOLA grouped TLP (GMM)", "NSGA-II (pymoo)"]
    budgets = [100, 200]
    return build_campaign_manifest(
        run_kind="multi_objective",
        budgets=budgets,
        n_runs=4,
        problem_names=[SYNTHETIC_GROUPED_TLP.name],
        optimizer_names=optimizers,
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [
                    {"budget": budget, "configuration": {"adapter": optimizer}}
                    for budget in budgets
                ],
            }
            for optimizer in optimizers
        ],
        provenance=TEST_PROVENANCE,
    )


def _write_grouped_campaign(path: Path) -> ResultStore:
    store = ResultStore(path)
    store.prepare_campaign(_grouped_manifest(), resume=False)
    gaps = [0.0, 0.25, 0.75, 1.0]
    for optimizer_index, optimizer in enumerate(("HOLA grouped TLP (GMM)", "NSGA-II (pymoo)")):
        for budget in (100, 200):
            for run_id in range(4):
                failed = optimizer_index == 1 and budget == 100 and run_id == 3
                front = [
                    [0.15 + 0.01 * run_id, 0.75],
                    [0.75, 0.15 + 0.01 * run_id],
                ]
                store.append_multi(
                    {
                        "problem": SYNTHETIC_GROUPED_TLP.name,
                        "optimizer": optimizer,
                        "budget": budget,
                        "run_id": run_id,
                        "seed": make_seed(SYNTHETIC_GROUPED_TLP.name, budget, run_id),
                        "status": "error" if failed else "success",
                        "error": "deliberate" if failed else "",
                        "optimizer_config": {"adapter": optimizer},
                        "n_evaluations": 50 if failed else budget,
                        "pareto_front": None if failed else front,
                        "decision_vectors": None if failed else [[0.0] * 5, [1.0] * 5],
                        "normalized_hypervolume_gap": (
                            None if failed else gaps[run_id] + 0.05 * optimizer_index
                        ),
                        "normalized_igd": (
                            None if failed else 0.1 + 0.01 * run_id + 0.02 * optimizer_index
                        ),
                        "spacing": None if failed else 0.1,
                        "wall_time_seconds": 0.1,
                        "n_pareto_points": None if failed else 2,
                    }
                )
    return store


def test_grouped_reporting_records_median_run_rule_and_every_plotted_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "plots"
    _write_grouped_campaign(results_dir)
    plotted_metrics: list[str] = []
    plotted_fronts: list[int] = []
    monkeypatch.setattr(
        grouped_reporting,
        "plot_grouped_metric",
        lambda summary, output, metric: plotted_metrics.append(metric),
    )
    monkeypatch.setattr(
        grouped_reporting,
        "plot_representative_fronts",
        lambda points, output: plotted_fronts.append(len(points)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot-grouped-tlp",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--bootstrap-resamples",
            "128",
        ],
    )

    grouped_reporting.main()

    summary = pd.read_csv(output_dir / "grouped_tlp_summary.csv")
    failures = pd.read_csv(output_dir / "grouped_tlp_failures.csv")
    selections = pd.read_csv(output_dir / "grouped_tlp_representative_front_selection.csv")
    points = pd.read_csv(output_dir / "grouped_tlp_representative_front_points.csv")
    assert set(summary["metric"]) == {
        "normalized_hypervolume_gap",
        "normalized_igd",
    }
    assert failures[["optimizer", "budget", "run_id"]].to_records(index=False).tolist() == [
        ("NSGA-II (pymoo)", 100, 3)
    ]
    assert set(selections["selection_rule"]) == {grouped_reporting.REPRESENTATIVE_SELECTION_RULE}
    assert set(selections["budget"]) == {200}
    assert set(selections["run_id"]) == {1}
    selected_points = points[points["source"] == "selected_method_run"]
    analytic_points = points[points["source"] == "analytic_pareto_front"]
    assert len(selected_points) == 4
    assert len(analytic_points) == len(SYNTHETIC_GROUPED_TLP.true_pareto_front)
    assert set(selected_points["run_id"]) == {1}
    assert plotted_metrics == ["normalized_hypervolume_gap", "normalized_igd"]
    assert plotted_fronts == [len(points)]


def test_representative_front_selection_rejects_malformed_stored_front() -> None:
    results = pd.DataFrame(
        [
            {
                "problem": SYNTHETIC_GROUPED_TLP.name,
                "optimizer": "optimizer",
                "budget": 100,
                "run_id": 0,
                "seed": 1,
                "status": "success",
                "normalized_hypervolume_gap": 0.1,
                "pareto_front": json.dumps([[0.1, 0.2, 0.3]]),
            }
        ]
    )
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        grouped_reporting.representative_front_tables(results)


def test_representative_front_selection_rejects_inconsistent_point_count() -> None:
    results = pd.DataFrame(
        [
            {
                "problem": SYNTHETIC_GROUPED_TLP.name,
                "optimizer": "optimizer",
                "budget": 100,
                "run_id": 0,
                "seed": 1,
                "status": "success",
                "normalized_hypervolume_gap": 0.1,
                "pareto_front": json.dumps([[0.1, 0.2], [0.2, 0.1]]),
                "n_pareto_points": 3,
            }
        ]
    )
    with pytest.raises(ValueError, match="point count"):
        grouped_reporting.representative_front_tables(results)


def test_secondary_plotting_renders_every_expected_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpo_results = _write_hpo_campaign(tmp_path / "hpo").load_complete_hpo()
    hpo_summaries = hpo_reporting.summarize_hpo(hpo_results, n_resamples=32)
    hpo_names: list[str] = []
    monkeypatch.setattr(
        hpo_reporting,
        "save_figure",
        lambda fig, output, name: hpo_names.append(name),
    )
    for metric, summary in hpo_summaries.items():
        hpo_reporting.plot_hpo_metric(summary, tmp_path, metric)

    grouped_results = _write_grouped_campaign(tmp_path / "grouped").load_complete_multi()
    grouped_summary = grouped_reporting.summarize_grouped_tlp(
        grouped_results,
        n_resamples=32,
    )
    _, grouped_points = grouped_reporting.representative_front_tables(grouped_results)
    grouped_names: list[str] = []
    monkeypatch.setattr(
        grouped_reporting,
        "save_figure",
        lambda fig, output, name: grouped_names.append(name),
    )
    for metric in grouped_reporting.GROUPED_METRICS:
        grouped_reporting.plot_grouped_metric(grouped_summary, tmp_path, metric)
    grouped_reporting.plot_representative_fronts(grouped_points, tmp_path)

    assert hpo_names == [
        "hpo_validation_r2_by_budget_gbr_diabetes_hpo",
        "hpo_heldout_test_r2_by_budget_gbr_diabetes_hpo",
    ]
    assert grouped_names == [
        f"grouped_tlp_normalized_hypervolume_gap_{SYNTHETIC_GROUPED_TLP.name}",
        f"grouped_tlp_normalized_igd_{SYNTHETIC_GROUPED_TLP.name}",
        f"grouped_tlp_representative_front_{SYNTHETIC_GROUPED_TLP.name}_200eval",
    ]
