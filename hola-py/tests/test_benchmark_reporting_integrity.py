# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Completeness and failure-awareness checks at reporting boundaries."""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pymoo")

import benchmarks.plotting.multi_objective as multi_plotting  # noqa: E402
import benchmarks.plotting.single_objective as single_plotting  # noqa: E402
from benchmarks.data.manifest import build_campaign_manifest  # noqa: E402
from benchmarks.data.persistence import ResultStore  # noqa: E402
from benchmarks.data.seeding import make_seed  # noqa: E402

pytestmark = pytest.mark.benchmarks


def _manifest(run_kind: str = "single_objective") -> dict[str, Any]:
    optimizers = ["reliable", "fragile"]
    return build_campaign_manifest(
        run_kind=run_kind,
        budgets=[10],
        n_runs=2,
        problem_names=["forrester_1d" if run_kind == "single_objective" else "zdt1_30d"],
        optimizer_names=optimizers,
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [
                    {
                        "budget": 10,
                        "configuration": {"adapter": optimizer, "budget": 10},
                    }
                ],
            }
            for optimizer in optimizers
        ],
    )


def _single_row(
    optimizer: str,
    run_id: int,
    *,
    status: str = "success",
    problem: str = "forrester_1d",
) -> dict[str, Any]:
    return {
        "problem": problem,
        "optimizer": optimizer,
        "budget": 10,
        "run_id": run_id,
        "seed": make_seed(problem, 10, run_id),
        "status": status,
        "error": "deliberate failure" if status == "error" else "",
        "optimizer_config": {"adapter": optimizer, "budget": 10},
        "n_evaluations": 10 if status == "success" else None,
        "best_value": -5.0 if status == "success" else None,
        "best_params": {"x": 0.75} if status == "success" else None,
        "wall_time_seconds": 0.1,
        "convergence_trace": [-5.0] * 10 if status == "success" else None,
    }


def _write_complete_single(store: ResultStore) -> None:
    store.prepare_campaign(_manifest(), resume=False)
    for optimizer in ("reliable", "fragile"):
        for run_id in range(2):
            status = "error" if optimizer == "fragile" and run_id == 1 else "success"
            store.append_single(_single_row(optimizer, run_id, status=status))


def _multi_row(optimizer: str, run_id: int, *, status: str) -> dict[str, Any]:
    return {
        "problem": "zdt1_30d",
        "optimizer": optimizer,
        "budget": 10,
        "run_id": run_id,
        "seed": make_seed("zdt1_30d", 10, run_id),
        "status": status,
        "error": "deliberate failure" if status == "error" else "",
        "optimizer_config": {"adapter": optimizer, "budget": 10},
        "n_evaluations": 10 if status == "success" else None,
        "pareto_front": [[0.0, 1.0]] if status == "success" else None,
        "decision_vectors": [[0.0] * 30] if status == "success" else None,
        "normalized_hypervolume_gap": 0.5 if status == "success" else None,
        "normalized_igd": 0.5 if status == "success" else None,
        "spacing": 0.1 if status == "success" else None,
        "wall_time_seconds": 0.1,
        "n_pareto_points": 1 if status == "success" else None,
    }


def _write_complete_multi(store: ResultStore) -> None:
    store.prepare_campaign(_manifest("multi_objective"), resume=False)
    for optimizer in ("reliable", "fragile"):
        for run_id in range(2):
            status = "error" if optimizer == "fragile" and run_id == 1 else "success"
            store.append_multi(_multi_row(optimizer, run_id, status=status))


def test_complete_campaign_accepts_error_rows_as_completed_outcomes(tmp_path) -> None:
    store = ResultStore(tmp_path)
    _write_complete_single(store)

    results = store.load_complete_single()

    assert len(results) == 4
    assert results["status"].value_counts().to_dict() == {"success": 3, "error": 1}

    multi_store = ResultStore(tmp_path / "multi")
    _write_complete_multi(multi_store)
    multi_results = multi_store.load_complete_multi()
    assert multi_results["status"].value_counts().to_dict() == {
        "success": 3,
        "error": 1,
    }


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("seed", "seed does not match deterministic derivation"),
        ("evaluations", "used 9 evaluations; expected 10"),
        ("trace", "convergence_trace has 9 entries; expected 10"),
        ("configuration", "optimizer_config does not match the manifest"),
    ],
)
def test_complete_single_rejects_result_contract_mismatches(
    tmp_path,
    violation: str,
    message: str,
) -> None:
    store = ResultStore(tmp_path)
    store.prepare_campaign(_manifest(), resume=False)
    rows = [
        _single_row(optimizer, run_id)
        for optimizer in ("reliable", "fragile")
        for run_id in range(2)
    ]
    target = rows[0]
    if violation == "seed":
        target["seed"] += 1
    elif violation == "evaluations":
        target["n_evaluations"] = 9
    elif violation == "trace":
        target["convergence_trace"] = [-5.0] * 9
    else:
        target["optimizer_config"] = {"adapter": "changed", "budget": 10}
    for row in rows:
        store.append_single(row)

    with pytest.raises(RuntimeError, match=message):
        store.load_complete_single()


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("seed", "seed does not match deterministic derivation"),
        ("evaluations", "used 9 evaluations; expected 10"),
        ("configuration", "optimizer_config does not match the manifest"),
    ],
)
def test_complete_multi_rejects_result_contract_mismatches(
    tmp_path,
    violation: str,
    message: str,
) -> None:
    store = ResultStore(tmp_path)
    store.prepare_campaign(_manifest("multi_objective"), resume=False)
    rows = [
        _multi_row(optimizer, run_id, status="success")
        for optimizer in ("reliable", "fragile")
        for run_id in range(2)
    ]
    target = rows[0]
    if violation == "seed":
        target["seed"] += 1
    elif violation == "evaluations":
        target["n_evaluations"] = 9
    else:
        target["optimizer_config"] = {"adapter": "changed", "budget": 10}
    for row in rows:
        store.append_multi(row)

    with pytest.raises(RuntimeError, match=message):
        store.load_complete_multi()


def _random_x2_manifest() -> dict[str, Any]:
    configuration = {
        "adapter": "RandomDoubleAdapter",
        "strategy": "HOLA random",
        "declared_budget": 10,
        "evaluation_multiplier": 2,
        "actual_budget": 20,
    }
    return build_campaign_manifest(
        run_kind="single_objective",
        budgets=[10],
        n_runs=1,
        problem_names=["forrester_1d"],
        optimizer_names=["Random x2"],
        optimizer_configurations=[
            {
                "optimizer": "Random x2",
                "by_budget": [{"budget": 10, "configuration": configuration}],
            }
        ],
    )


@pytest.mark.parametrize(("n_evaluations", "accepted"), [(20, True), (10, False)])
def test_random_x2_reporting_requires_twice_the_declared_budget(
    tmp_path,
    n_evaluations: int,
    accepted: bool,
) -> None:
    store = ResultStore(tmp_path)
    manifest = _random_x2_manifest()
    configuration = manifest["optimizer_configurations"][0]["by_budget"][0]["configuration"]
    store.prepare_campaign(manifest, resume=False)
    row = _single_row("Random x2", 0)
    row["optimizer_config"] = configuration
    row["n_evaluations"] = n_evaluations
    row["convergence_trace"] = [-5.0] * n_evaluations
    store.append_single(row)

    if accepted:
        assert len(store.load_complete_single()) == 1
    else:
        with pytest.raises(RuntimeError, match="used 10 evaluations; expected 20"):
            store.load_complete_single()


@pytest.mark.parametrize("violation", ["partial", "duplicate", "extra", "bad_status"])
def test_complete_campaign_refuses_invalid_row_coverage(tmp_path, violation: str) -> None:
    store = ResultStore(tmp_path)
    store.prepare_campaign(_manifest(), resume=False)
    rows = [
        _single_row(optimizer, run_id)
        for optimizer in ("reliable", "fragile")
        for run_id in range(2)
    ]
    if violation == "partial":
        rows.pop()
        expected = "incomplete"
    elif violation == "duplicate":
        rows.append(dict(rows[0]))
        expected = "duplicate run keys"
    elif violation == "extra":
        rows[-1] = _single_row("fragile", 2)
        expected = "unexpected run key"
    else:
        rows[-1]["status"] = "failed"
        expected = "malformed statuses"
    for row in rows:
        store.append_single(row)

    with pytest.raises(RuntimeError, match=expected):
        store.load_complete_single()


def test_reporting_refuses_tampered_manifest_and_wrong_run_kind(tmp_path) -> None:
    tampered = ResultStore(tmp_path / "tampered")
    _write_complete_single(tampered)
    payload = json.loads(tampered.manifest_path.read_text())
    payload["budgets"] = [999]
    tampered.manifest_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="fingerprint"):
        tampered.load_complete_single()

    wrong_kind = ResultStore(tmp_path / "wrong-kind")
    wrong_kind.prepare_campaign(_manifest("multi_objective"), resume=False)
    with pytest.raises(RuntimeError, match="expected a single_objective campaign"):
        wrong_kind.load_complete_single()


def test_single_objective_reporting_validates_before_output(tmp_path, monkeypatch) -> None:
    results_dir = tmp_path / "partial"
    output_dir = tmp_path / "plots"
    store = ResultStore(results_dir)
    store.prepare_campaign(_manifest(), resume=False)
    store.append_single(_single_row("reliable", 0))
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

    with pytest.raises(RuntimeError, match="incomplete"):
        single_plotting.main()

    assert not output_dir.exists()


def test_single_objective_reporting_writes_explicit_failure_table(
    tmp_path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "complete"
    output_dir = tmp_path / "plots"
    _write_complete_single(ResultStore(results_dir))
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

    failures = pd.read_csv(output_dir / "single_objective_failures.csv")
    summary = pd.read_csv(output_dir / "single_objective_regret_summary.csv")
    assert failures[["optimizer", "run_id"]].to_records(index=False).tolist() == [("fragile", 1)]
    fragile = summary[summary["optimizer"] == "fragile"].iloc[0]
    assert fragile["n_total_runs"] == 2
    assert fragile["n_successful_runs"] == 1
    assert fragile["n_failed_runs"] == 1
    assert fragile["success_rate"] == 0.5


def test_multi_objective_reporting_writes_explicit_failure_table(
    tmp_path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "complete"
    output_dir = tmp_path / "plots"
    _write_complete_multi(ResultStore(results_dir))
    monkeypatch.setattr(multi_plotting, "plot_metric_by_budget", lambda *args: None)
    monkeypatch.setattr(
        multi_plotting,
        "plot_family_balanced_metric_ranks",
        lambda *args: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot-multi",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    multi_plotting.main()

    failures = pd.read_csv(output_dir / "multi_objective_failures.csv")
    summary = pd.read_csv(output_dir / "multi_objective_summary.csv")
    assert failures[["optimizer", "run_id"]].to_records(index=False).tolist() == [("fragile", 1)]
    fragile = summary[summary["optimizer"] == "fragile"].iloc[0]
    assert fragile["n_total_runs"] == 2
    assert fragile["n_successful_runs"] == 1
    assert fragile["n_failed_runs"] == 1
    assert fragile["success_rate"] == 0.5
