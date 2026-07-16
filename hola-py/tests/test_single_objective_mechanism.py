# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for the paired single-objective mechanism comparison."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import benchmarks.plotting.single_objective as single_plotting  # noqa: E402
from benchmarks.data.manifest import build_campaign_manifest  # noqa: E402
from benchmarks.data.normalize import (  # noqa: E402
    GMM_MECHANISM_COMPARATORS,
    GMM_OPTIMIZER,
    aggregate_family_balanced_paired_win_rates,
    paired_win_outcomes,
)
from benchmarks.data.persistence import ResultStore  # noqa: E402
from benchmarks.data.seeding import make_seed  # noqa: E402

pytestmark = pytest.mark.benchmarks


def _analysis_row(
    *,
    family: str,
    problem: str,
    dimension: int,
    optimizer: str,
    run_id: int,
    regret: float | None,
    status: str = "success",
    budget: int = 10,
) -> dict[str, object]:
    return {
        "suite": "synthetic",
        "family": family,
        "problem": problem,
        "dimension": dimension,
        "optimizer": optimizer,
        "budget": budget,
        "run_id": run_id,
        "status": status,
        "regret": regret,
    }


def test_paired_outcomes_join_by_run_id_not_input_order() -> None:
    rows = [
        _analysis_row(
            family="one",
            problem="one_2d",
            dimension=2,
            optimizer=GMM_OPTIMIZER,
            run_id=0,
            regret=1.0,
        ),
        _analysis_row(
            family="one",
            problem="one_2d",
            dimension=2,
            optimizer=GMM_OPTIMIZER,
            run_id=1,
            regret=3.0,
        ),
        _analysis_row(
            family="one",
            problem="one_2d",
            dimension=2,
            optimizer="baseline",
            run_id=1,
            regret=2.0,
        ),
        _analysis_row(
            family="one",
            problem="one_2d",
            dimension=2,
            optimizer="baseline",
            run_id=0,
            regret=2.0,
        ),
    ]

    outcomes = paired_win_outcomes(pd.DataFrame(rows), comparators=("baseline",))

    assert outcomes[["run_id", "outcome"]].to_records(index=False).tolist() == [
        (0, 1.0),
        (1, 0.0),
    ]

    with pytest.raises(ValueError, match="unpaired runs"):
        paired_win_outcomes(pd.DataFrame(rows[:-1]), comparators=("baseline",))


def test_family_balance_averages_dimensions_before_families() -> None:
    rows = []
    task_outcomes = [
        ("multi", "multi_2d", 2, 1.0),
        ("multi", "multi_7d", 7, 0.0),
        ("single", "single_2d", 2, 1.0),
    ]
    for family, problem, dimension, outcome in task_outcomes:
        for run_id in range(3):
            focal_regret, baseline_regret = (0.0, 1.0) if outcome == 1.0 else (1.0, 0.0)
            rows.extend(
                [
                    _analysis_row(
                        family=family,
                        problem=problem,
                        dimension=dimension,
                        optimizer=GMM_OPTIMIZER,
                        run_id=run_id,
                        regret=focal_regret,
                    ),
                    _analysis_row(
                        family=family,
                        problem=problem,
                        dimension=dimension,
                        optimizer="baseline",
                        run_id=run_id,
                        regret=baseline_regret,
                    ),
                ]
            )

    summary = aggregate_family_balanced_paired_win_rates(
        pd.DataFrame(rows),
        comparators=("baseline",),
        n_bootstrap=100,
    ).iloc[0]

    assert summary["win_rate"] == pytest.approx(0.75)
    assert summary["ci_lower"] == pytest.approx(0.75)
    assert summary["ci_upper"] == pytest.approx(0.75)
    assert summary["n_families"] == 2
    assert summary["n_tasks"] == 3
    assert summary["n_paired_runs"] == 3
    assert summary["n_paired_outcomes"] == 9


def test_paired_failures_and_exact_metric_ties_have_explicit_scores() -> None:
    rows = []
    cases = [
        ("success", 1.0, "error", None, 1.0),
        ("error", None, "success", 1.0, 0.0),
        ("error", None, "error", None, 0.5),
        ("success", 1.0, "success", 1.0, 0.5),
    ]
    for run_id, (focal_status, focal_regret, other_status, other_regret, _) in enumerate(cases):
        rows.extend(
            [
                _analysis_row(
                    family="one",
                    problem="one_2d",
                    dimension=2,
                    optimizer=GMM_OPTIMIZER,
                    run_id=run_id,
                    regret=focal_regret,
                    status=focal_status,
                ),
                _analysis_row(
                    family="one",
                    problem="one_2d",
                    dimension=2,
                    optimizer="baseline",
                    run_id=run_id,
                    regret=other_regret,
                    status=other_status,
                ),
            ]
        )

    outcomes = paired_win_outcomes(pd.DataFrame(rows), comparators=("baseline",))

    assert outcomes["outcome"].tolist() == [case[-1] for case in cases]


def test_paired_bootstrap_is_deterministic_and_row_order_invariant() -> None:
    rows = []
    for run_id, focal_regret in enumerate([0.0, 2.0, 0.0, 2.0, 0.0]):
        for family in ("left", "right"):
            rows.extend(
                [
                    _analysis_row(
                        family=family,
                        problem=f"{family}_2d",
                        dimension=2,
                        optimizer=GMM_OPTIMIZER,
                        run_id=run_id,
                        regret=focal_regret,
                    ),
                    _analysis_row(
                        family=family,
                        problem=f"{family}_2d",
                        dimension=2,
                        optimizer="baseline",
                        run_id=run_id,
                        regret=1.0,
                    ),
                ]
            )
    frame = pd.DataFrame(rows)

    first = aggregate_family_balanced_paired_win_rates(
        frame,
        comparators=("baseline",),
        n_bootstrap=257,
        seed=42,
    )
    second = aggregate_family_balanced_paired_win_rates(
        frame.sample(frac=1.0, random_state=9),
        comparators=("baseline",),
        n_bootstrap=257,
        seed=42,
    )

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert first["ci_lower"].item() < first["win_rate"].item()
    assert first["ci_upper"].item() > first["win_rate"].item()


def test_paired_bootstrap_ignores_run_id_alignment_between_tasks() -> None:
    rows = []
    outcomes_by_problem = {
        "left_2d": [1.0, 1.0, 0.0, 0.0],
        "right_2d": [0.0, 0.0, 1.0, 1.0],
    }
    for problem, outcomes in outcomes_by_problem.items():
        family = problem.removesuffix("_2d")
        for run_id, outcome in enumerate(outcomes):
            focal_regret, baseline_regret = (0.0, 1.0) if outcome else (1.0, 0.0)
            rows.extend(
                [
                    _analysis_row(
                        family=family,
                        problem=problem,
                        dimension=2,
                        optimizer=GMM_OPTIMIZER,
                        run_id=run_id,
                        regret=focal_regret,
                    ),
                    _analysis_row(
                        family=family,
                        problem=problem,
                        dimension=2,
                        optimizer="baseline",
                        run_id=run_id,
                        regret=baseline_regret,
                    ),
                ]
            )
    frame = pd.DataFrame(rows)
    permuted = frame.copy()
    right = permuted["problem"].eq("right_2d")
    permuted.loc[right, "run_id"] = permuted.loc[right, "run_id"].map({0: 2, 1: 3, 2: 0, 3: 1})

    original_summary = aggregate_family_balanced_paired_win_rates(
        frame,
        comparators=("baseline",),
        n_bootstrap=1024,
        seed=73,
    )
    permuted_summary = aggregate_family_balanced_paired_win_rates(
        permuted,
        comparators=("baseline",),
        n_bootstrap=1024,
        seed=73,
    )

    pd.testing.assert_frame_equal(original_summary, permuted_summary)
    assert original_summary["win_rate"].item() == pytest.approx(0.5)
    assert original_summary["ci_lower"].item() < 0.5
    assert original_summary["ci_upper"].item() > 0.5


def _campaign_manifest() -> dict[str, Any]:
    optimizers = [GMM_OPTIMIZER, *GMM_MECHANISM_COMPARATORS]
    return build_campaign_manifest(
        run_kind="single_objective",
        budgets=[10],
        n_runs=2,
        problem_names=["forrester_1d"],
        optimizer_names=optimizers,
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [{"budget": 10, "configuration": _campaign_configuration(optimizer)}],
            }
            for optimizer in optimizers
        ],
    )


def _campaign_configuration(optimizer: str) -> dict[str, object]:
    configuration: dict[str, object] = {"adapter": optimizer}
    if optimizer == "Random x2":
        configuration.update(
            {
                "declared_budget": 10,
                "evaluation_multiplier": 2,
                "actual_budget": 20,
            }
        )
    return configuration


def _campaign_row(optimizer: str, run_id: int) -> dict[str, object]:
    value = {
        GMM_OPTIMIZER: -6.0,
        "HOLA (random)": -5.0,
        "HOLA (sobol)": -5.5,
        "Random x2": -5.75,
    }[optimizer]
    return {
        "problem": "forrester_1d",
        "optimizer": optimizer,
        "budget": 10,
        "run_id": run_id,
        "seed": make_seed("forrester_1d", 10, run_id),
        "status": "success",
        "error": "",
        "optimizer_config": _campaign_configuration(optimizer),
        "n_evaluations": 20 if optimizer == "Random x2" else 10,
        "best_value": value,
        "best_params": {"x": 0.75},
        "wall_time_seconds": 0.1,
        "convergence_trace": [value] * (20 if optimizer == "Random x2" else 10),
    }


def test_plot_single_emits_mechanism_csv_and_paper_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "plots"
    store = ResultStore(results_dir)
    store.prepare_campaign(_campaign_manifest(), resume=False)
    for optimizer in [GMM_OPTIMIZER, *GMM_MECHANISM_COMPARATORS]:
        for run_id in range(2):
            store.append_single(_campaign_row(optimizer, run_id))

    saved: list[str] = []
    monkeypatch.setattr(
        single_plotting,
        "save_figure",
        lambda _fig, _path, name: saved.append(name),
    )
    monkeypatch.setattr(single_plotting, "plot_family_balanced_ranks", lambda *args: None)
    monkeypatch.setattr(single_plotting, "plot_regret_by_family", lambda *args: None)
    monkeypatch.setattr(single_plotting, "plot_box_per_benchmark", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot-single",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    single_plotting.main()

    output = pd.read_csv(output_dir / "single_objective_gmm_paired_win_rates.csv")
    assert output["comparator"].tolist() == sorted(GMM_MECHANISM_COMPARATORS)
    assert output["win_rate"].tolist() == [1.0, 1.0, 1.0]
    assert saved == ["gmm_family_balanced_paired_win_rate"]
