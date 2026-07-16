# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Protocol tests for the explicitly grouped-TLP capability problem."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("pymoo")

from benchmarks.adapters.hola_adapter import HolaGroupedTlpAdapter  # noqa: E402
from benchmarks.functions.grouped_tlp import (  # noqa: E402
    synthetic_grouped_tlp,
    synthetic_grouped_tlp_pareto_front,
)
from benchmarks.problems.grouped_tlp import (  # noqa: E402
    GROUPED_TLP_PROBLEMS,
    SYNTHETIC_GROUPED_TLP,
)

pytestmark = pytest.mark.benchmarks


def _params(*values: float) -> dict[str, float]:
    return {f"x{index}": value for index, value in enumerate(values)}


def test_group_schema_is_explicit_and_priorities_only_vary_within_groups() -> None:
    problem = SYNTHETIC_GROUPED_TLP

    assert list(GROUPED_TLP_PROBLEMS) == ["synthetic_grouped_tlp_5d"]
    assert problem.group_names == ("group_a", "group_b")
    assert problem.n_groups == 2
    assert [objective.target for objective in problem.objectives] == pytest.approx(
        [0.2, 0.04, 0.2, 0.04]
    )
    assert [objective.limit for objective in problem.objectives] == pytest.approx(
        [0.9, 0.81, 0.9, 0.81]
    )
    grouped_priorities = {
        group: [objective.priority for objective in problem.objectives if objective.group == group]
        for group in problem.group_names
    }
    assert grouped_priorities == {"group_a": [1.0, 2.0], "group_b": [2.0, 1.0]}


def test_raw_objectives_and_shared_group_transform_are_checkable() -> None:
    problem = SYNTHETIC_GROUPED_TLP
    raw_at_origin = synthetic_grouped_tlp(_params(0.0, 0.0, 0.0, 0.0, 0.0))

    assert raw_at_origin == {"f1": 0.0, "f2": 0.0, "f3": 1.0, "f4": 1.0}
    origin_costs = problem.group_costs(raw_at_origin)
    assert origin_costs["group_a"] == 0.0
    assert np.isinf(origin_costs["group_b"])

    left_endpoint = _params(0.2, 0.0, 0.0, 0.0, 0.0)
    right_endpoint = _params(0.8, 0.0, 0.0, 0.0, 0.0)
    assert problem.evaluate_group_costs(left_endpoint) == pytest.approx(
        {"group_a": 0.0, "group_b": 192.0 / 77.0}
    )
    assert problem.evaluate_group_costs(right_endpoint) == pytest.approx(
        {"group_a": 186.0 / 77.0, "group_b": 0.0}
    )

    center = problem.evaluate_group_costs(_params(0.5, 0.0, 0.0, 0.0, 0.0))
    assert center == pytest.approx({"group_a": 75.0 / 77.0, "group_b": 87.0 / 77.0})

    limit_violation = problem.evaluate_group_costs(_params(0.85, 1.0, 1.0, 0.0, 0.0))
    assert np.isinf(limit_violation["group_a"])
    assert np.isfinite(limit_violation["group_b"])


def test_stored_group_cost_front_matches_closed_form_geometry() -> None:
    problem = SYNTHETIC_GROUPED_TLP
    expected = synthetic_grouped_tlp_pareto_front(len(problem.true_pareto_front))

    np.testing.assert_array_equal(problem.true_pareto_front, expected)
    np.testing.assert_allclose(problem.true_pareto_front[0], [0.0, 192.0 / 77.0])
    np.testing.assert_allclose(problem.true_pareto_front[-1], [186.0 / 77.0, 0.0])
    assert np.all(np.diff(problem.true_pareto_front[:, 0]) > 0.0)
    assert np.all(np.diff(problem.true_pareto_front[:, 1]) < 0.0)
    assert problem.ideal_point == (0.0, 0.0)
    assert problem.reference_point == (3.0, 3.0)
    assert problem.as_multi_objective_problem().normalized_reference_hypervolume == pytest.approx(
        4325.0 / 5929.0
    )
    for index, tradeoff in enumerate(np.linspace(0.2, 0.8, 5)):
        params = _params(tradeoff, 0.0, 0.0, 0.0, 0.0)
        costs = problem.evaluate_group_costs(params)
        expected_index = index * (len(problem.true_pareto_front) - 1) // 4
        np.testing.assert_allclose(
            [costs[group] for group in problem.group_names],
            problem.true_pareto_front[expected_index],
            rtol=0.0,
            atol=1e-15,
        )


def test_target_plateaus_and_nuisance_penalty_are_dominated_by_the_exact_set() -> None:
    problem = SYNTHETIC_GROUPED_TLP

    def vector(params: dict[str, float]) -> np.ndarray:
        costs = problem.evaluate_group_costs(params)
        return np.asarray([costs[group] for group in problem.group_names])

    def dominates(left: np.ndarray, right: np.ndarray) -> bool:
        return bool(np.all(left <= right) and np.any(left < right))

    assert dominates(
        vector(_params(0.2, 0.0, 0.0, 0.0, 0.0)),
        vector(_params(0.15, 0.0, 0.0, 0.0, 0.0)),
    )
    assert dominates(
        vector(_params(0.8, 0.0, 0.0, 0.0, 0.0)),
        vector(_params(0.85, 0.0, 0.0, 0.0, 0.0)),
    )
    assert dominates(
        vector(_params(0.5, 0.0, 0.0, 0.0, 0.0)),
        vector(_params(0.5, 1.0, 0.0, 0.0, 0.0)),
    )


def test_competitor_problem_uses_the_shared_transform_and_is_picklable() -> None:
    problem = SYNTHETIC_GROUPED_TLP
    competitor_problem = problem.as_multi_objective_problem()
    params = _params(0.15, 0.75, 0.1, 0.2, 0.3)
    expected = problem.group_costs(problem.raw_func(params))

    assert competitor_problem.objective_names == problem.group_names
    assert competitor_problem.func(params) == pytest.approx(expected)
    infeasible = _params(0.0, 0.0, 0.0, 0.0, 0.0)
    assert competitor_problem.infeasible_sentinel == (4.0, 4.0)
    assert tuple(competitor_problem.func(infeasible).values()) == (4.0, 4.0)
    assert competitor_problem.is_infeasible_objectives((4.0, 4.0))
    restored = pickle.loads(pickle.dumps(competitor_problem))
    assert restored.func(params) == pytest.approx(expected)


def test_hola_grouped_adapter_returns_the_same_group_cost_transform() -> None:
    problem = SYNTHETIC_GROUPED_TLP
    competitor_problem = problem.as_multi_objective_problem()
    result = HolaGroupedTlpAdapter(problem, strategy="gmm").optimize(
        competitor_problem,
        budget=24,
        seed=123,
    )

    assert result.n_evaluations == 24
    assert result.decision_vectors is not None
    assert len(result.pareto_front) == len(result.decision_vectors)
    parameter_names = list(problem.bounds)
    for costs, decision in zip(result.pareto_front, result.decision_vectors, strict=True):
        params = dict(zip(parameter_names, decision, strict=True))
        expected = problem.evaluate_group_costs(params)
        np.testing.assert_allclose(
            costs,
            [expected[group] for group in problem.group_names],
            rtol=0.0,
            atol=1e-12,
        )
