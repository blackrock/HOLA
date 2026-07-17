# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused checks for audited representative multi-objective fronts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pymoo")

import benchmarks.plotting.multi_objective as multi_plotting  # noqa: E402

pytestmark = pytest.mark.benchmarks


def _result_row(
    optimizer: str,
    run_id: int,
    gap: float | None,
    *,
    problem: str = "zdt3_30d",
    budget: int = 200,
    status: str = "success",
    igd: float | None = None,
    front: Any = None,
    n_pareto_points: int | None = None,
) -> dict[str, Any]:
    if front is None and status == "success":
        front = json.dumps([[0.1, 0.9], [0.8, 0.2]])
    if n_pareto_points is None and status == "success":
        n_pareto_points = 2
    return {
        "problem": problem,
        "optimizer": optimizer,
        "budget": budget,
        "run_id": run_id,
        "seed": 10_000 + run_id,
        "status": status,
        "error": "deliberate failure" if status == "error" else "",
        "n_evaluations": budget if status == "success" else 17,
        "pareto_front": front,
        "normalized_hypervolume_gap": gap,
        "normalized_igd": None if gap is None else (gap + 0.1 if igd is None else igd),
        "spacing": None if gap is None else 0.05,
        "wall_time_seconds": 0.25,
        "n_pareto_points": n_pareto_points,
    }


def test_parse_stored_objective_front_accepts_numeric_json_and_rejects_bad_data() -> None:
    parsed = multi_plotting.parse_stored_objective_front(
        json.dumps([[0.0, 1.0], [1.0, 0.0]]),
        2,
    )
    np.testing.assert_array_equal(parsed, [[0.0, 1.0], [1.0, 0.0]])

    invalid = [
        ("not JSON", "valid JSON"),
        (json.dumps([]), r"shape \(n, 2\)"),
        (json.dumps([0.0, 1.0]), r"shape \(n, 2\)"),
        (json.dumps([[0.0, 1.0, 2.0]]), r"shape \(n, 2\)"),
        (json.dumps([[0.0, float("nan")]]), "finite"),
        ({"not": "a front"}, "numeric"),
    ]
    for value, message in invalid:
        with pytest.raises(ValueError, match=message):
            multi_plotting.parse_stored_objective_front(value, 2)


def test_representative_selection_uses_largest_budget_median_and_run_id_tie_break() -> None:
    results = pd.DataFrame(
        [
            _result_row("alpha", 9, 0.0, budget=100),
            _result_row("alpha", 2, 0.1),
            _result_row("alpha", 0, 0.3),
            _result_row("alpha", 5, None, status="error"),
            _result_row("beta", 7, 0.4),
            _result_row("beta", 1, 0.2),
        ]
    )

    selections, points = multi_plotting.representative_terminal_front_tables(
        results,
        problem_names=("zdt3_30d",),
    )

    assert selections[["optimizer", "run_id"]].to_records(index=False).tolist() == [
        ("alpha", 0),
        ("beta", 1),
    ]
    alpha = selections[selections["optimizer"].eq("alpha")].iloc[0]
    assert alpha["budget"] == 200
    assert alpha["source_result_row_index"] == 2
    assert alpha["method_median_hypervolume_gap"] == pytest.approx(0.2)
    assert alpha["absolute_distance_to_median"] == pytest.approx(0.1)
    assert alpha["n_failed_runs_at_budget"] == 1
    assert alpha["failure_errors"] == "deliberate failure"
    assert alpha["selection_rule"] == multi_plotting.REPRESENTATIVE_SELECTION_RULE
    assert set(points["run_id"]) == {0, 1}
    assert len(points) == 4


def test_representative_selection_uses_igd_before_run_id_for_hv_ties() -> None:
    results = pd.DataFrame(
        [
            _result_row("alpha", 0, 1.0, igd=0.9),
            _result_row("alpha", 1, 1.0, igd=0.2),
            _result_row("alpha", 2, 1.0, igd=0.4),
        ]
    )

    selections, _ = multi_plotting.representative_terminal_front_tables(
        results,
        problem_names=("zdt3_30d",),
    )

    selected = selections.iloc[0]
    assert selected["run_id"] == 2
    assert selected["method_median_igd"] == pytest.approx(0.4)
    assert selected["absolute_igd_distance_to_median"] == pytest.approx(0.0)


def test_invalid_and_missing_fronts_are_audited_and_annotated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = pd.DataFrame(
        [
            _result_row(
                "broken",
                0,
                0.1,
                front=json.dumps([[0.1, 0.2, 0.3]]),
                n_pareto_points=1,
            ),
            _result_row("failed", 1, None, status="error"),
            _result_row("failed", 2, float("nan")),
            _result_row("mixed", 3, 0.2),
            _result_row("mixed", 4, None, status="error"),
        ]
    )
    selections, points = multi_plotting.representative_terminal_front_tables(
        results,
        problem_names=("zdt3_30d",),
    )

    statuses = selections.set_index("optimizer")["selection_status"].to_dict()
    assert statuses == {
        "broken": "invalid_stored_front",
        "failed": "no_successful_finite_hv_run",
        "mixed": "selected",
    }
    broken = selections[selections["optimizer"].eq("broken")].iloc[0]
    assert broken["run_id"] == 0
    assert "shape (n, 2)" in broken["front_validation_error"]
    assert set(points["optimizer"]) == {"mixed"}

    captured: dict[str, Any] = {}

    def capture_figure(figure: Any, output_dir: Path, name: str) -> None:
        captured.update(figure=figure, output_dir=output_dir, name=name)

    monkeypatch.setattr(multi_plotting, "save_figure", capture_figure)
    multi_plotting.plot_representative_terminal_fronts(selections, points, tmp_path)

    assert captured["name"] == "multi_objective_representative_terminal_fronts"
    assert captured["output_dir"] == tmp_path
    ax = captured["figure"].axes[0]
    annotation = "\n".join(text.get_text() for text in ax.texts)
    assert "broken: selected front invalid" in annotation
    assert "failed: no successful finite-HV run; 1 failed run" in annotation
    assert "mixed: 1 failed run" in annotation
    labels = ax.get_legend_handles_labels()[1]
    assert "Fixed true Pareto front" in labels
    assert "Reporting reference point" in labels
    assert "mixed" in labels


def test_plot_multi_writes_selected_row_and_metrics_to_audit_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = pd.DataFrame(
        [
            _result_row("alpha", 1, 0.3),
            _result_row("alpha", 0, 0.1),
        ]
    )
    monkeypatch.setattr(
        multi_plotting,
        "load_reporting_results",
        lambda *_: results,
    )
    monkeypatch.setattr(multi_plotting, "plot_metric_by_budget", lambda *args: None)
    monkeypatch.setattr(
        multi_plotting,
        "plot_family_balanced_metric_ranks",
        lambda *args: None,
    )
    plotted: list[tuple[int, int]] = []
    monkeypatch.setattr(
        multi_plotting,
        "plot_representative_terminal_fronts",
        lambda selections, points, _: plotted.append((len(selections), len(points))),
    )
    output_dir = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot-multi", "--results-dir", str(tmp_path), "--output-dir", str(output_dir)],
    )

    multi_plotting.main()

    audit = pd.read_csv(output_dir / "multi_objective_representative_front_selection.csv")
    assert audit.loc[0, "run_id"] == 0
    assert audit.loc[0, "source_result_row_index"] == 1
    assert audit.loc[0, "normalized_hypervolume_gap"] == pytest.approx(0.1)
    assert audit.loc[0, "normalized_igd"] == pytest.approx(0.2)
    assert audit.loc[0, "spacing"] == pytest.approx(0.05)
    assert plotted == [(1, 2)]
