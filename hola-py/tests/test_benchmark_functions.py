# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Reference checks for the analytic benchmark objectives and registered optima."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pymoo.problems import get_problem

from benchmarks.problems.multi_objective import MULTI_OBJECTIVE_PROBLEMS
from benchmarks.problems.single_objective import SINGLE_OBJECTIVE_PROBLEMS

pytestmark = pytest.mark.benchmarks


@pytest.mark.parametrize(
    ("problem_name", "params"),
    [
        ("forrester_1d", {"x": 0.757248757841856}),
        ("branin_2d", {"x1": -math.pi, "x2": 12.275}),
        ("bukin6_2d", {"x1": -10.0, "x2": 1.0}),
        ("cross_in_tray_2d", {"x1": 1.34940668535334, "x2": 1.34940668535334}),
        ("drop_wave_2d", {"x1": 0.0, "x2": 0.0}),
        ("egg_holder_2d", {"x1": 512.0, "x2": 404.231805123}),
        ("holder_table_2d", {"x1": 8.055023472141116, "x2": 9.664590028909654}),
        ("levy13_2d", {"x1": 1.0, "x2": 1.0}),
        (
            "six_hump_camel_2d",
            {"x1": 0.08984201368301331, "x2": -0.7126564032704135},
        ),
    ],
)
def test_registered_closed_form_minima_match_known_minimizers(
    problem_name: str,
    params: dict[str, float],
) -> None:
    problem = SINGLE_OBJECTIVE_PROBLEMS[problem_name]
    assert problem.func(params) == pytest.approx(problem.known_minimum, abs=1e-12, rel=0.0)


@pytest.mark.parametrize("family", ["ackley", "rastrigin"])
@pytest.mark.parametrize("dimension", [2, 5, 7])
def test_zero_minimum_nd_families(family: str, dimension: int) -> None:
    problem = SINGLE_OBJECTIVE_PROBLEMS[f"{family}_{dimension}d"]
    params = {f"x{index}": 0.0 for index in range(dimension)}
    assert problem.func(params) == pytest.approx(problem.known_minimum, abs=1e-12, rel=0.0)


@pytest.mark.parametrize("dimension", [2, 5, 7])
def test_schwefel_registry_matches_the_rounded_implementation(dimension: int) -> None:
    problem = SINGLE_OBJECTIVE_PROBLEMS[f"schwefel_{dimension}d"]
    minimizer = 420.9687463599821
    params = {f"x{index}": minimizer for index in range(dimension)}
    assert problem.func(params) == pytest.approx(problem.known_minimum, abs=3e-13, rel=0.0)
    assert problem.known_minimum > 0.0


@pytest.mark.parametrize(
    "problem_name",
    [
        name
        for name, problem in MULTI_OBJECTIVE_PROBLEMS.items()
        if problem.family.startswith(("zdt", "dtlz"))
    ],
)
def test_zdt_and_dtlz_objectives_match_pymoo(problem_name: str) -> None:
    problem = MULTI_OBJECTIVE_PROBLEMS[problem_name]
    kwargs = {"n_var": problem.dimensionality}
    if problem.family.startswith("dtlz"):
        kwargs["n_obj"] = problem.n_objectives
    reference = get_problem(problem.family, **kwargs)

    rng = np.random.default_rng(20260715)
    lower = np.asarray([bounds[0] for bounds in problem.bounds.values()])
    upper = np.asarray([bounds[1] for bounds in problem.bounds.values()])
    for _ in range(3):
        decision = lower + (upper - lower) * rng.random(problem.dimensionality)
        params = dict(zip(problem.bounds, decision, strict=True))
        observed = problem.func(params)
        actual = np.asarray([observed[name] for name in problem.objective_names])
        expected = np.asarray(reference.evaluate(decision[None, :]))[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
