# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for the sealed-test practical HPO benchmark protocol."""

from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import optuna
import pytest

import benchmarks.adapters.hpo as hpo_adapter_module
from benchmarks.adapters.base import EmpiricalExploitationError, HpoOptimizationResult
from benchmarks.adapters.hpo import (
    HolaHpoAdapter,
    OptunaTpeHpoAdapter,
    build_hola_parameter_map,
    suggest_optuna_params,
)
from benchmarks.data.manifest import build_campaign_manifest
from benchmarks.data.persistence import ResultStore
from benchmarks.data.seeding import make_hpo_split_seed, make_seed
from benchmarks.problems.hpo import (
    DIABETES_GBR_HPO,
    CategoricalParameter,
    HpoProblem,
    IntegerParameter,
    RealParameter,
)
from benchmarks.runner import run_hpo as hpo_runner
from benchmarks.runner.config import RunConfig
from benchmarks.runner.run_hpo import _run_hpo_one
from hola_opt import Categorical, Integer, Real

pytestmark = pytest.mark.benchmarks


class RecordingEvaluator:
    def __init__(self, problem: HpoProblem, validation_budget: int) -> None:
        self.problem = problem
        self.validation_budget = validation_budget
        self.validation_calls = 0
        self.heldout_calls = 0
        self.params_seen: list[dict[str, Any]] = []
        self.events: list[str] = []

    @property
    def split_sizes(self) -> tuple[int, int, int]:
        return 6, 2, 2

    def evaluate_validation(self, params: dict[str, Any]) -> float:
        if self.validation_calls >= self.validation_budget:
            raise RuntimeError("validation budget exhausted")
        normalized = self.problem.normalize_params(params)
        self.validation_calls += 1
        self.params_seen.append(normalized)
        self.events.append("validation")
        return float(normalized["n_estimators"]) + float(normalized["learning_rate"])

    def evaluate_heldout(self, params: dict[str, Any]) -> float:
        if self.validation_calls != self.validation_budget:
            raise RuntimeError("heldout before validation completion")
        if self.heldout_calls:
            raise RuntimeError("heldout called twice")
        self.problem.normalize_params(params)
        self.heldout_calls += 1
        self.events.append("heldout")
        return 0.42


def _toy_problem(evaluator: RecordingEvaluator | None = None) -> HpoProblem:
    parameters = {
        "n_estimators": IntegerParameter(1, 5),
        "max_depth": IntegerParameter(1, 3),
        "learning_rate": RealParameter(1e-3, 0.1, scale="log10"),
        "subsample": RealParameter(0.5, 1.0),
        "loss": CategoricalParameter(("squared_error", "huber")),
    }
    holder: dict[str, RecordingEvaluator | None] = {"evaluator": evaluator}

    def factory(problem: HpoProblem, split_seed: int, budget: int) -> RecordingEvaluator:
        del split_seed
        if holder["evaluator"] is None:
            holder["evaluator"] = RecordingEvaluator(problem, budget)
        result = holder["evaluator"]
        assert result is not None
        return result

    return HpoProblem("toy_hpo", parameters, factory)


class FakeOptunaTrial:
    def __init__(self) -> None:
        self.calls: dict[str, tuple[Any, ...]] = {}

    def suggest_int(self, name: str, minimum: int, maximum: int) -> int:
        self.calls[name] = ("integer", minimum, maximum)
        return minimum

    def suggest_float(self, name: str, minimum: float, maximum: float, *, log: bool) -> float:
        self.calls[name] = ("real", minimum, maximum, log)
        return minimum

    def suggest_categorical(self, name: str, choices: list[str]) -> str:
        self.calls[name] = ("categorical", tuple(choices))
        return choices[0]


def test_native_space_translation_preserves_types_scales_and_choices() -> None:
    problem = _toy_problem()
    hola = build_hola_parameter_map(problem)
    assert isinstance(hola["n_estimators"], Integer)
    assert isinstance(hola["max_depth"], Integer)
    assert isinstance(hola["learning_rate"], Real)
    assert hola["learning_rate"].scale == "log10"
    assert isinstance(hola["subsample"], Real)
    assert hola["subsample"].scale == "linear"
    assert isinstance(hola["loss"], Categorical)
    assert hola["loss"].choices == ["squared_error", "huber"]

    fake = FakeOptunaTrial()
    suggest_optuna_params(problem, cast(optuna.Trial, fake))
    assert fake.calls == {
        "n_estimators": ("integer", 1, 5),
        "max_depth": ("integer", 1, 3),
        "learning_rate": ("real", 1e-3, 0.1, True),
        "subsample": ("real", 0.5, 1.0, False),
        "loss": ("categorical", ("squared_error", "huber")),
    }


@pytest.mark.parametrize(
    "adapter",
    [
        HolaHpoAdapter("random"),
        HolaHpoAdapter("sobol"),
        OptunaTpeHpoAdapter(),
    ],
    ids=lambda adapter: adapter.name,
)
def test_native_hpo_adapters_use_exact_validation_budget(adapter: object) -> None:
    problem = _toy_problem()
    evaluator = RecordingEvaluator(problem, validation_budget=5)
    result = adapter.optimize(  # type: ignore[attr-defined]
        problem,
        evaluator.evaluate_validation,
        budget=5,
        seed=123,
    )
    assert result.n_evaluations == evaluator.validation_calls == 5
    assert len(result.validation_trace) == 5
    assert evaluator.heldout_calls == 0
    assert isinstance(result.best_params["n_estimators"], int)
    assert isinstance(result.best_params["max_depth"], int)
    assert result.best_params["loss"] in {"squared_error", "huber"}


def test_gmm_gate_failure_preserves_validation_count_and_never_calls_heldout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = RecordingEvaluator(_toy_problem(), validation_budget=5)
    problem = _toy_problem(evaluator)
    evaluator.problem = problem

    def fail_gate(study: object, completed_evaluations: int) -> None:
        del study
        raise EmpiricalExploitationError(
            completed_evaluations,
            {
                "completed_evaluations": completed_evaluations,
                "gmm_fit_epoch": 1,
                "gmm_origin_suggestions": 4,
                "gmm_sampling_ready": True,
                "issued_suggestions": completed_evaluations,
            },
        )

    monkeypatch.setattr(
        hpo_adapter_module,
        "require_empirical_gmm_exploitation",
        fail_gate,
    )

    row = _run_hpo_one(problem, HolaHpoAdapter("gmm"), budget=5, run_id=0)

    assert row["status"] == "error"
    assert row["n_validation_evaluations"] == 5
    assert row["n_heldout_evaluations"] == 0
    assert "EmpiricalExploitationError" in row["error"]
    assert "gmm_origin_suggestions" in row["error"]
    assert evaluator.events == ["validation"] * 5


def test_runner_calls_heldout_once_only_after_exact_validation_budget() -> None:
    evaluator = RecordingEvaluator(_toy_problem(), validation_budget=3)
    problem = _toy_problem(evaluator)
    evaluator.problem = problem

    class CompleteOptimizer:
        name = "complete"

        def configuration(self, budget: int) -> dict[str, object]:
            return {"adapter": type(self).__name__, "budget": budget}

        def optimize(self, problem, evaluate_validation, budget, seed):
            del seed
            params = {
                "n_estimators": 2,
                "max_depth": 1,
                "learning_rate": 0.01,
                "subsample": 1.0,
                "loss": "squared_error",
            }
            trace = [evaluate_validation(params) for _ in range(budget)]
            return HpoOptimizationResult(max(trace), params, 0.0, budget, trace)

    row = _run_hpo_one(problem, CompleteOptimizer(), budget=3, run_id=0)
    assert row["status"] == "success"
    assert row["n_validation_evaluations"] == 3
    assert row["n_heldout_evaluations"] == 1
    assert row["heldout_test_r2"] == 0.42
    assert row["search_seed"] == make_seed(problem.name, 3, 0)
    assert row["split_seed"] == make_hpo_split_seed(problem.name, 0)
    assert evaluator.events == ["validation", "validation", "validation", "heldout"]


def test_hpo_split_seed_is_fixed_across_budgets_while_search_seed_is_horizon_specific() -> None:
    base_problem = _toy_problem()
    observed_split_seeds: list[tuple[int, int]] = []

    def factory(problem: HpoProblem, split_seed: int, budget: int) -> RecordingEvaluator:
        observed_split_seeds.append((budget, split_seed))
        return RecordingEvaluator(problem, budget)

    problem = HpoProblem("seeded_toy_hpo", base_problem.parameters, factory)
    observed_search_seeds: list[tuple[str, int, int]] = []

    class CompleteOptimizer:
        def __init__(self, name: str) -> None:
            self.name = name

        def configuration(self, budget: int) -> dict[str, object]:
            return {"adapter": type(self).__name__, "budget": budget}

        def optimize(self, problem, evaluate_validation, budget, seed):
            observed_search_seeds.append((self.name, budget, seed))
            params = {
                "n_estimators": 2,
                "max_depth": 1,
                "learning_rate": 0.01,
                "subsample": 1.0,
                "loss": "squared_error",
            }
            trace = [evaluate_validation(params) for _ in range(budget)]
            return HpoOptimizationResult(max(trace), params, 0.0, budget, trace)

    rows = [
        _run_hpo_one(problem, CompleteOptimizer("left"), budget=2, run_id=7),
        _run_hpo_one(problem, CompleteOptimizer("left"), budget=3, run_id=7),
        _run_hpo_one(problem, CompleteOptimizer("right"), budget=2, run_id=7),
    ]
    assert all(row["status"] == "success" for row in rows)
    assert rows[0]["split_seed"] == rows[1]["split_seed"] == rows[2]["split_seed"]
    assert rows[0]["search_seed"] == rows[2]["search_seed"]
    assert rows[0]["search_seed"] != rows[1]["search_seed"]
    assert {split_seed for _, split_seed in observed_split_seeds} == {rows[0]["split_seed"]}
    assert observed_search_seeds == [
        ("left", 2, rows[0]["search_seed"]),
        ("left", 3, rows[1]["search_seed"]),
        ("right", 2, rows[2]["search_seed"]),
    ]


def test_hpo_campaign_manifest_records_distinct_seed_scopes(tmp_path: Path) -> None:
    problem = _toy_problem()

    class CompleteOptimizer:
        name = "complete"

        def configuration(self, budget: int) -> dict[str, object]:
            return {"adapter": type(self).__name__, "budget": budget}

        def optimize(self, problem, evaluate_validation, budget, seed):
            del problem, seed
            params = {
                "n_estimators": 2,
                "max_depth": 1,
                "learning_rate": 0.01,
                "subsample": 1.0,
                "loss": "squared_error",
            }
            trace = [evaluate_validation(params) for _ in range(budget)]
            return HpoOptimizationResult(max(trace), params, 0.0, budget, trace)

    hpo_runner.run_hpo(
        [problem],
        [CompleteOptimizer()],
        RunConfig(
            output_dir=tmp_path,
            n_runs=1,
            n_workers=1,
            budgets=[1],
            resume=False,
        ),
    )
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    seeds = manifest["campaign_configuration"]["seed_derivation"]
    assert seeds == {
        "search_seed": "first 32 bits of SHA-256(problem:budget:run_id)",
        "split_seed": "first 32 bits of SHA-256(hpo-split:problem:run_id)",
        "pairing": (
            "search seed paired across optimizers within problem+budget+run_id; "
            "split seed paired across optimizers and fixed across budgets within "
            "problem+run_id"
        ),
    }
    split_config = manifest["campaign_configuration"]["problems"][0]["split"]
    assert split_config["split_seed_scope"] == (
        "problem+run_id; fixed across budgets and optimizers"
    )


def test_failed_incomplete_optimization_never_calls_heldout() -> None:
    evaluator = RecordingEvaluator(_toy_problem(), validation_budget=3)
    problem = _toy_problem(evaluator)
    evaluator.problem = problem

    class IncompleteOptimizer:
        name = "incomplete"

        def optimize(self, problem, evaluate_validation, budget, seed):
            del problem, seed
            params = {
                "n_estimators": 2,
                "max_depth": 1,
                "learning_rate": 0.01,
                "subsample": 1.0,
                "loss": "squared_error",
            }
            trace = [evaluate_validation(params) for _ in range(budget - 1)]
            return HpoOptimizationResult(max(trace), params, 0.0, budget - 1, trace)

    row = _run_hpo_one(problem, IncompleteOptimizer(), budget=3, run_id=0)
    assert row["status"] == "error"
    assert row["n_validation_evaluations"] == 2
    assert row["n_heldout_evaluations"] == 0
    assert evaluator.events == ["validation", "validation"]


def test_diabetes_split_is_deterministic_and_heldout_is_sealed() -> None:
    first = DIABETES_GBR_HPO.make_evaluator(split_seed=123, validation_budget=1)
    second = DIABETES_GBR_HPO.make_evaluator(split_seed=123, validation_budget=1)
    params = {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "loss": "squared_error",
    }
    with pytest.raises(RuntimeError, match="only after"):
        first.evaluate_heldout(params)
    assert first.heldout_calls == 0
    assert first.split_sizes == second.split_sizes
    assert first.evaluate_validation(params) == pytest.approx(second.evaluate_validation(params))
    heldout = first.evaluate_heldout(params)
    assert isinstance(heldout, float)
    assert first.heldout_calls == 1
    with pytest.raises(RuntimeError, match="only once"):
        first.evaluate_heldout(params)


def test_hpo_work_item_is_pickleable_and_runs_in_real_process_pool() -> None:
    pickle.dumps((DIABETES_GBR_HPO, HolaHpoAdapter("random"), 1, 0))
    with ProcessPoolExecutor(max_workers=1) as executor:
        row = executor.submit(
            _run_hpo_one,
            DIABETES_GBR_HPO,
            HolaHpoAdapter("random"),
            1,
            0,
        ).result(timeout=60)
    assert row["status"] == "success", row["error"]
    assert row["n_validation_evaluations"] == 1
    assert row["n_heldout_evaluations"] == 1


def _test_manifest() -> dict[str, Any]:
    return build_campaign_manifest(
        run_kind="hpo",
        budgets=[25],
        n_runs=1,
        problem_names=[DIABETES_GBR_HPO.name],
        optimizer_names=["optimizer"],
        optimizer_configurations=[{"optimizer": "optimizer", "by_budget": []}],
        campaign_configuration={"problems": [DIABETES_GBR_HPO.configuration()]},
        provenance={
            "code": {"commit": "test", "dirty": False, "source_hash": "test"},
            "lock_hash": "test",
            "python": {"implementation": "CPython", "version": "test"},
            "platform": {"platform": "test", "machine": "test", "system": "test"},
            "dependencies": {"hola-opt": None},
            "native_extension": None,
        },
    )


def test_hpo_has_dedicated_persistence_and_resume_key(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    store.prepare_campaign(_test_manifest(), resume=False)
    store.append_hpo(
        {
            "problem": DIABETES_GBR_HPO.name,
            "optimizer": "optimizer",
            "budget": 25,
            "run_id": 0,
            "search_seed": 123,
            "split_seed": 456,
            "status": "success",
            "error": "",
            "optimizer_config": {"adapter": "test"},
            "n_validation_evaluations": 25,
            "best_validation_r2": 0.5,
            "best_params": {"n_estimators": 50, "loss": "huber"},
            "validation_trace": [0.1, 0.5],
            "heldout_test_r2": 0.4,
            "n_heldout_evaluations": 1,
            "train_size": 264,
            "validation_size": 89,
            "test_size": 89,
            "wall_time_seconds": 1.0,
        }
    )
    assert store.hpo_path.exists()
    assert not store.so_path.exists()
    row = store.load_hpo().iloc[0]
    assert row["search_seed"] == 123
    assert row["split_seed"] == 456
    assert json.loads(row["best_params"]) == {"loss": "huber", "n_estimators": 50}
    assert json.loads(row["validation_trace"]) == [0.1, 0.5]
    assert store.completed_hpo_runs() == {(DIABETES_GBR_HPO.name, "optimizer", 25, 0)}


def _paired_seed_manifest() -> dict[str, Any]:
    budgets = [25, 50]
    return build_campaign_manifest(
        run_kind="hpo",
        budgets=budgets,
        n_runs=1,
        problem_names=[DIABETES_GBR_HPO.name],
        optimizer_names=["left", "right"],
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [
                    {"budget": budget, "configuration": {"adapter": "test"}} for budget in budgets
                ],
            }
            for optimizer in ("left", "right")
        ],
        provenance=_test_manifest()["provenance"],
    )


def _paired_seed_row(optimizer: str, budget: int) -> dict[str, Any]:
    return {
        "problem": DIABETES_GBR_HPO.name,
        "optimizer": optimizer,
        "budget": budget,
        "run_id": 0,
        "search_seed": make_seed(DIABETES_GBR_HPO.name, budget, 0),
        "split_seed": make_hpo_split_seed(DIABETES_GBR_HPO.name, 0),
        "status": "success",
        "error": "",
        "optimizer_config": {"adapter": "test"},
        "n_validation_evaluations": budget,
        "best_validation_r2": 0.5,
        "best_params": {"n_estimators": 50, "loss": "huber"},
        "validation_trace": [0.5] * budget,
        "heldout_test_r2": 0.4,
        "n_heldout_evaluations": 1,
        "train_size": 264,
        "validation_size": 89,
        "test_size": 89,
        "wall_time_seconds": 1.0,
    }


def _write_seed_pairing_campaign(
    path: Path,
    *,
    violation: str | None = None,
) -> ResultStore:
    store = ResultStore(path)
    store.prepare_campaign(_paired_seed_manifest(), resume=False)
    for optimizer in ("left", "right"):
        for budget in (25, 50):
            row = _paired_seed_row(optimizer, budget)
            if violation == "search" and optimizer == "right" and budget == 25:
                row["search_seed"] += 1
            if violation == "split" and budget == 50:
                row["split_seed"] += 1
            if violation == "validation" and optimizer == "right" and budget == 25:
                row["n_validation_evaluations"] = 24
                row["n_heldout_evaluations"] = 0
                row["validation_trace"] = [0.5] * 24
            if violation == "heldout" and optimizer == "right" and budget == 25:
                row["n_heldout_evaluations"] = 0
            if violation == "trace" and optimizer == "right" and budget == 25:
                row["validation_trace"] = [0.5] * 24
            if violation == "configuration" and optimizer == "right" and budget == 25:
                row["optimizer_config"] = {"adapter": "changed"}
            store.append_hpo(row)
    return store


def test_complete_hpo_campaign_accepts_distinct_documented_seed_scopes(tmp_path: Path) -> None:
    rows = _write_seed_pairing_campaign(tmp_path).load_complete_hpo()
    assert len(rows) == 4
    assert rows.groupby(["problem", "budget", "run_id"])["search_seed"].nunique().eq(1).all()
    assert rows.groupby(["problem", "run_id"])["split_seed"].nunique().eq(1).all()


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("search", "search_seed does not match deterministic derivation"),
        ("split", "split_seed does not match deterministic derivation"),
    ],
)
def test_complete_hpo_campaign_rejects_broken_seed_derivation(
    tmp_path: Path,
    violation: str,
    message: str,
) -> None:
    store = _write_seed_pairing_campaign(tmp_path, violation=violation)
    with pytest.raises(RuntimeError, match=message):
        store.load_complete_hpo()


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("validation", "used 24 validation evaluations; expected 25"),
        ("heldout", "must use exactly one held-out evaluation"),
        ("trace", "validation_trace length does not match its calls"),
        ("configuration", "optimizer_config does not match the manifest"),
    ],
)
def test_complete_hpo_campaign_rejects_result_contract_mismatches(
    tmp_path: Path,
    violation: str,
    message: str,
) -> None:
    store = _write_seed_pairing_campaign(tmp_path, violation=violation)
    with pytest.raises(RuntimeError, match=message):
        store.load_complete_hpo()


def test_hpo_runner_defaults_to_preregistered_budgets_and_native_optimizers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def capture(problems, optimizers, config) -> None:
        captured.update(problems=problems, optimizers=optimizers, config=config)

    monkeypatch.setattr(hpo_runner, "run_hpo", capture)
    monkeypatch.setattr(sys, "argv", ["run-hpo", "--output-dir", str(tmp_path)])
    hpo_runner.main()
    assert captured["config"].budgets == [25, 50, 100]
    assert captured["config"].n_runs == 1
    assert [problem.name for problem in captured["problems"]] == [DIABETES_GBR_HPO.name]
    assert [optimizer.name for optimizer in captured["optimizers"]] == [
        "HOLA HPO (random)",
        "HOLA HPO (sobol)",
        "HOLA HPO (GMM)",
        "Optuna HPO (TPE)",
    ]
