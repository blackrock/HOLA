# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for benchmark fairness and evaluation-budget contracts."""

from __future__ import annotations

import json
import pickle
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("pymoo")
pytest.importorskip("optuna")

from benchmarks.adapters.base import (  # noqa: E402
    EvaluationCountError,
    MultiObjectiveResult,
    SingleObjectiveResult,
)
from benchmarks.adapters.hola_adapter import (  # noqa: E402
    HolaGroupedTlpAdapter,
    HolaMultiObjectiveAdapter,
    HolaSingleObjectiveAdapter,
)
from benchmarks.adapters.igr_adapter import IGRAdapter  # noqa: E402
from benchmarks.adapters.optuna_adapter import (  # noqa: E402
    OptunaNSGAIIAdapter,
    OptunaTPEAdapter,
)
from benchmarks.adapters.pymoo_multi import (  # noqa: E402
    PymooMOEADAdapter,
    PymooNSGAIIAdapter,
)
from benchmarks.adapters.pymoo_single import (  # noqa: E402
    ga_adapter,
    hooke_jeeves_adapter,
    nelder_mead_adapter,
    pso_adapter,
)
from benchmarks.adapters.random_double import RandomDoubleAdapter  # noqa: E402
from benchmarks.data.persistence import ResultStore  # noqa: E402
from benchmarks.data.seeding import make_seed  # noqa: E402
from benchmarks.problems.grouped_tlp import SYNTHETIC_GROUPED_TLP  # noqa: E402
from benchmarks.problems.registry import (  # noqa: E402
    MultiObjectiveProblem,
    SingleObjectiveProblem,
)
from benchmarks.problems.single_objective import SINGLE_OBJECTIVE_PROBLEMS  # noqa: E402
from benchmarks.runner.config import RunConfig  # noqa: E402
from benchmarks.runner.executor import (  # noqa: E402
    _bounded_futures,
    _run_multi_one,
    _run_single_one,
    run_single_objective,
)


@pytest.fixture
def scalar_problem() -> SingleObjectiveProblem:
    return SingleObjectiveProblem(
        name="quadratic_1d",
        func=lambda params: (params["x"] - 0.25) ** 2,
        bounds={"x": (0.0, 1.0)},
        known_minimum=0.0,
    )


@pytest.fixture
def vector_problem() -> MultiObjectiveProblem:
    def objective(params: dict[str, float]) -> dict[str, float]:
        x = params["x"]
        return {"left": x**2, "right": (1.0 - x) ** 2}

    return MultiObjectiveProblem(
        name="tradeoff_1d",
        func=objective,
        bounds={"x": (0.0, 1.0)},
        objective_names=("left", "right"),
        reference_point=(2.0, 2.0),
        ideal_point=(0.0, 0.0),
        true_pareto_front=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )


@pytest.mark.benchmarks
def test_seed_is_paired_across_optimizers() -> None:
    seed = make_seed("problem", budget=100, run_id=7)
    assert seed == make_seed("problem", budget=100, run_id=7)
    assert seed != make_seed("problem", budget=100, run_id=8)


@pytest.mark.benchmarks
def test_executor_pairs_seed_across_optimizer_names_and_records_actual_count(
    scalar_problem: SingleObjectiveProblem,
) -> None:
    class RecordingOptimizer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.seed: int | None = None

        def optimize(
            self, problem: SingleObjectiveProblem, budget: int, seed: int
        ) -> SingleObjectiveResult:
            self.seed = seed
            return SingleObjectiveResult(
                best_value=0.0,
                best_params={"x": 0.25},
                wall_time_seconds=0.0,
                n_evaluations=budget,
                convergence_trace=[0.0] * budget,
            )

    left = RecordingOptimizer("left")
    right = RecordingOptimizer("right")
    left_row = _run_single_one(scalar_problem, left, budget=7, run_id=3)
    right_row = _run_single_one(scalar_problem, right, budget=7, run_id=3)

    assert left.seed == right.seed == left_row["seed"] == right_row["seed"]
    assert left_row["n_evaluations"] == right_row["n_evaluations"] == 7
    assert left_row["status"] == right_row["status"] == "success"
    assert left_row["best_params"] == {"x": 0.25}
    assert left_row["optimizer_config"]["adapter"] == "RecordingOptimizer"


@pytest.mark.benchmarks
def test_executor_captures_run_failures_with_actual_evaluation_count(
    scalar_problem: SingleObjectiveProblem,
) -> None:
    class FailingOptimizer:
        name = "failing"

        def configuration(self, budget: int) -> dict[str, object]:
            return {"adapter": type(self).__name__, "budget": budget}

        def optimize(
            self, problem: SingleObjectiveProblem, budget: int, seed: int
        ) -> SingleObjectiveResult:
            raise EvaluationCountError(actual=3, expected=budget, optimizer=self.name)

    row = _run_single_one(scalar_problem, FailingOptimizer(), budget=7, run_id=0)

    assert row["status"] == "error"
    assert row["n_evaluations"] == 3
    assert row["optimizer_config"] == {"adapter": "FailingOptimizer", "budget": 7}
    assert "EvaluationCountError" in row["error"]


@pytest.mark.benchmarks
def test_campaign_persists_failure_and_continues_to_next_run(
    tmp_path,
    scalar_problem: SingleObjectiveProblem,
) -> None:
    class SuccessfulOptimizer:
        name = "successful"

        def configuration(self, budget: int) -> dict[str, object]:
            return {"adapter": type(self).__name__, "budget": budget, "mode": "success"}

        def optimize(
            self, problem: SingleObjectiveProblem, budget: int, seed: int
        ) -> SingleObjectiveResult:
            return SingleObjectiveResult(
                best_value=0.0,
                best_params={"x": 0.25},
                wall_time_seconds=0.0,
                n_evaluations=budget,
                convergence_trace=[0.0] * budget,
            )

    class FailingOptimizer:
        name = "failing"

        def configuration(self, budget: int) -> dict[str, object]:
            return {"adapter": type(self).__name__, "budget": budget, "mode": "failure"}

        def optimize(
            self, problem: SingleObjectiveProblem, budget: int, seed: int
        ) -> SingleObjectiveResult:
            raise RuntimeError("deliberate failure")

    run_single_objective(
        [scalar_problem],
        [FailingOptimizer(), SuccessfulOptimizer()],
        RunConfig(output_dir=tmp_path, budgets=[2], n_runs=1, n_workers=1, resume=False),
    )

    rows = {row["optimizer"]: row for row in ResultStore(tmp_path).load_single().to_dict("records")}
    assert set(rows) == {"failing", "successful"}
    assert rows["failing"]["status"] == "error"
    assert "deliberate failure" in rows["failing"]["error"]
    assert rows["successful"]["status"] == "success"
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    frozen = {
        entry["optimizer"]: entry["by_budget"] for entry in manifest["optimizer_configurations"]
    }
    assert frozen == {
        "failing": [
            {
                "budget": 2,
                "configuration": {
                    "adapter": "FailingOptimizer",
                    "budget": 2,
                    "mode": "failure",
                },
            }
        ],
        "successful": [
            {
                "budget": 2,
                "configuration": {
                    "adapter": "SuccessfulOptimizer",
                    "budget": 2,
                    "mode": "success",
                },
            }
        ],
    }


@pytest.mark.benchmarks
def test_parallel_submission_is_bounded() -> None:
    class ImmediateExecutor:
        def __init__(self) -> None:
            self.n_submitted = 0

        def submit(self, function, *args):
            self.n_submitted += 1
            future: Future[dict[str, Any]] = Future()
            future.set_result(function(*args))
            return future

    executor = ImmediateExecutor()
    work: list[tuple[Any, ...]] = [(index,) for index in range(10)]
    results = _bounded_futures(
        cast(Executor, executor),
        lambda index: {"index": index},
        work,
        max_in_flight=3,
    )

    first_future, _ = next(results)
    assert executor.n_submitted == 3
    observed = [first_future.result(), *(future.result() for future, _ in results)]
    assert len(observed) == 10
    assert executor.n_submitted == 10


@pytest.mark.benchmarks
def test_multiobjective_executor_records_raw_front_and_decisions(
    vector_problem: MultiObjectiveProblem,
) -> None:
    class RecordingOptimizer:
        name = "recording"

        def configuration(self, budget: int) -> dict[str, object]:
            return {"adapter": type(self).__name__, "budget": budget}

        def optimize(
            self, problem: MultiObjectiveProblem, budget: int, seed: int
        ) -> MultiObjectiveResult:
            return MultiObjectiveResult(
                pareto_front=np.array([[0.0, 1.0], [1.0, 0.0]]),
                decision_vectors=np.array([[0.0], [1.0]]),
                wall_time_seconds=0.0,
                n_evaluations=budget,
            )

    row = _run_multi_one(vector_problem, RecordingOptimizer(), budget=2, run_id=0)

    assert row["status"] == "success"
    np.testing.assert_array_equal(row["pareto_front"], [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(row["decision_vectors"], [[0.0], [1.0]])
    assert row["normalized_hypervolume_gap"] == pytest.approx(0.0)
    assert row["normalized_igd"] == pytest.approx(0.0)


@pytest.mark.benchmarks
def test_multiobjective_executor_captures_run_failure(
    vector_problem: MultiObjectiveProblem,
) -> None:
    class FailingOptimizer:
        name = "failing"

        def optimize(
            self, problem: MultiObjectiveProblem, budget: int, seed: int
        ) -> MultiObjectiveResult:
            raise RuntimeError("deliberate multi-objective failure")

    row = _run_multi_one(vector_problem, FailingOptimizer(), budget=2, run_id=0)

    assert row["status"] == "error"
    assert "deliberate multi-objective failure" in row["error"]
    assert row["pareto_front"] is None
    assert row["decision_vectors"] is None


@pytest.mark.benchmarks
def test_hola_multiobjective_uses_unbounded_raw_objectives(
    vector_problem: MultiObjectiveProblem,
) -> None:
    objectives = HolaMultiObjectiveAdapter._build_objectives(vector_problem)
    assert [objective.field for objective in objectives] == ["left", "right"]
    assert all(objective.target is None for objective in objectives)
    assert all(objective.limit is None for objective in objectives)


@pytest.mark.benchmarks
def test_hola_grouped_tlp_uses_explicit_groups_and_priorities() -> None:
    problem = SYNTHETIC_GROUPED_TLP
    objectives = HolaGroupedTlpAdapter._build_objectives(problem)

    assert [objective.field for objective in objectives] == ["f1", "f2", "f3", "f4"]
    assert [objective.group for objective in objectives] == [
        "group_a",
        "group_a",
        "group_b",
        "group_b",
    ]
    assert [objective.priority for objective in objectives] == [1.0, 2.0, 2.0, 1.0]


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "adapter",
    [
        HolaSingleObjectiveAdapter("gmm"),
        IGRAdapter(),
        OptunaTPEAdapter(),
        ga_adapter(),
        pso_adapter(),
        hooke_jeeves_adapter(),
    ],
    ids=lambda adapter: adapter.name,
)
def test_single_objective_adapters_use_exact_budget(
    adapter: object,
    scalar_problem: SingleObjectiveProblem,
) -> None:
    budget = 27
    result = adapter.optimize(scalar_problem, budget=budget, seed=123)  # type: ignore[attr-defined]
    assert result.n_evaluations == budget
    assert len(result.convergence_trace) == budget


@pytest.mark.benchmarks
def test_primary_pymoo_single_adapters_run_through_a_real_process_pool() -> None:
    """Campaign workers must be able to pickle every primary adapter and work item."""
    problem = SINGLE_OBJECTIVE_PROBLEMS["forrester_1d"]
    adapters = [
        ga_adapter(),
        pso_adapter(),
        hooke_jeeves_adapter(),
    ]
    for adapter in adapters:
        pickle.loads(pickle.dumps(adapter))

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_run_single_one, problem, adapter, 25, index)
            for index, adapter in enumerate(adapters)
        ]
        rows = [future.result(timeout=60) for future in futures]

    assert [row["optimizer"] for row in rows] == [adapter.name for adapter in adapters]
    assert all(row["status"] == "success" for row in rows)
    assert all(row["n_evaluations"] == 25 for row in rows)


@pytest.mark.benchmarks
def test_nelder_mead_adapter_is_explicitly_non_primary() -> None:
    adapter = nelder_mead_adapter()
    assert adapter.configuration(25)["protocol_role"] == "non-primary diagnostic"


@pytest.mark.benchmarks
def test_random_x2_reports_its_calibration_budget(
    scalar_problem: SingleObjectiveProblem,
) -> None:
    result = RandomDoubleAdapter().optimize(scalar_problem, budget=13, seed=123)
    assert result.n_evaluations == 26
    assert len(result.convergence_trace) == 26


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "adapter",
    [
        HolaMultiObjectiveAdapter("random"),
        OptunaNSGAIIAdapter(),
        PymooNSGAIIAdapter(),
        PymooMOEADAdapter(),
    ],
    ids=lambda adapter: adapter.name,
)
def test_multiobjective_adapters_use_exact_budget(
    adapter: object,
    vector_problem: MultiObjectiveProblem,
) -> None:
    budget = 12
    result = adapter.optimize(vector_problem, budget=budget, seed=123)  # type: ignore[attr-defined]
    assert result.n_evaluations == budget
    assert result.pareto_front.shape[1] == vector_problem.n_objectives


@pytest.mark.benchmarks
def test_pso_fails_loudly_when_no_valid_swarm_divides_budget(
    scalar_problem: SingleObjectiveProblem,
) -> None:
    with pytest.raises(ValueError, match="population-size divisor"):
        pso_adapter().optimize(scalar_problem, budget=29, seed=123)
