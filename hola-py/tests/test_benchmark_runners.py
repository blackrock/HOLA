# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for benchmark entry-point protocol defaults and filtering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytest

import benchmarks.runner.run_grouped_tlp as grouped_runner
import benchmarks.runner.run_multi_objective as multi_runner
import benchmarks.runner.run_single_objective as single_runner

pytestmark = pytest.mark.benchmarks


def _arguments(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "output_dir": tmp_path,
        "n_runs": 1,
        "n_workers": 1,
        "budgets": "25",
        "problems": None,
        "optimizers": None,
        "no_resume": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_single_objective_primary_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def capture(problems, optimizers, config) -> None:
        captured.update(problems=problems, optimizers=optimizers, config=config)

    monkeypatch.setattr(single_runner, "run_single_objective", capture)
    monkeypatch.setattr(sys, "argv", ["run-single"])

    single_runner.main()

    config = captured["config"]
    problems = captured["problems"]
    optimizers = captured["optimizers"]
    assert config.budgets == [200, 500, 1000, 2000]
    assert problems
    assert all(problem.suite == "synthetic" for problem in problems)
    assert "gbr_diabetes" not in {problem.name for problem in problems}
    optimizer_names = [optimizer.name for optimizer in optimizers]
    assert "IGR" not in optimizer_names
    assert "Nelder-Mead" not in optimizer_names
    assert "Random x2" in optimizer_names


def test_single_objective_explicit_all_and_optimizer_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def capture(problems, optimizers, config) -> None:
        captured.update(problems=problems, optimizers=optimizers, config=config)

    monkeypatch.setattr(single_runner, "run_single_objective", capture)
    single_runner.main(
        _arguments(
            tmp_path,
            problems="all",
            optimizers="TPE,HOLA (GMM),Random x2",
        )
    )

    assert "gbr_diabetes" not in {problem.name for problem in captured["problems"]}
    assert [optimizer.name for optimizer in captured["optimizers"]] == [
        "TPE",
        "HOLA (GMM)",
        "Random x2",
    ]


def test_multi_objective_optimizer_filter_preserves_requested_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def capture(problems, optimizers, config) -> None:
        captured.update(problems=problems, optimizers=optimizers, config=config)

    monkeypatch.setattr(multi_runner, "run_multi_objective", capture)
    multi_runner.main(
        _arguments(
            tmp_path,
            budgets="100",
            optimizers="MOEA/D,HOLA MO (sobol),NSGA-II (Optuna)",
        )
    )

    assert [optimizer.name for optimizer in captured["optimizers"]] == [
        "MOEA/D",
        "HOLA MO (sobol)",
        "NSGA-II (Optuna)",
    ]


def test_grouped_tlp_runner_uses_one_transformed_problem_and_fixed_optimizers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def capture(problems, optimizers, config) -> None:
        captured.update(problems=problems, optimizers=optimizers, config=config)

    monkeypatch.setattr(grouped_runner, "run_multi_objective", capture)
    grouped_runner.main(
        _arguments(
            tmp_path,
            budgets="100,200",
            optimizers=None,
        )
    )

    assert [problem.name for problem in captured["problems"]] == ["synthetic_grouped_tlp_5d"]
    assert captured["problems"][0].objective_names == ("group_a", "group_b")
    assert [optimizer.name for optimizer in captured["optimizers"]] == [
        "HOLA grouped TLP (GMM)",
        "NSGA-II (Optuna)",
        "NSGA-II (pymoo)",
        "MOEA/D",
    ]
    assert captured["config"].budgets == [100, 200]


@pytest.mark.parametrize(
    ("runner", "entrypoint", "valid_name"),
    [
        (single_runner, "run_single_objective", "Random x2"),
        (multi_runner, "run_multi_objective", "MOEA/D"),
    ],
)
def test_unknown_optimizer_name_has_actionable_cli_error(
    runner,
    entrypoint: str,
    valid_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner, entrypoint, lambda *args: pytest.fail("executor was called"))

    with pytest.raises(SystemExit) as raised:
        runner.main(_arguments(tmp_path, optimizers="not-an-optimizer"))

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "unknown optimizer name(s): not-an-optimizer" in error
    assert "Available names:" in error
    assert valid_name in error
