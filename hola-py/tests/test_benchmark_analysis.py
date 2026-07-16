# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for fixed-scale benchmark analysis and reporting."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("pymoo")
pytest.importorskip("scipy")
from scipy.integrate import quad  # noqa: E402

from benchmarks.data.normalize import (  # noqa: E402
    REGRET_TOLERANCE,
    add_simple_regret,
    aggregate_family_balanced_ranks,
    lexicographic_failure_ranks,
    summarize_regret,
)
from benchmarks.functions import zdt  # noqa: E402
from benchmarks.metrics.hypervolume import compute_normalized_hv_gap  # noqa: E402
from benchmarks.metrics.igd import compute_normalized_igd  # noqa: E402
from benchmarks.metrics.normalization import normalize_objectives  # noqa: E402
from benchmarks.metrics.spacing import (  # noqa: E402
    compute_normalized_spacing,
    compute_spacing,
)
from benchmarks.plotting.multi_objective import (  # noqa: E402
    aggregate_family_balanced_metric_ranks,
    summarize_multiobjective_metrics,
)
from benchmarks.problems.multi_objective import MULTI_OBJECTIVE_PROBLEMS  # noqa: E402
from benchmarks.problems.registry import MultiObjectiveProblem  # noqa: E402
from benchmarks.problems.single_objective import SINGLE_OBJECTIVE_PROBLEMS  # noqa: E402


@pytest.mark.benchmarks
def test_regret_uses_fixed_known_optimum_and_rejects_material_undershoot() -> None:
    problem = SINGLE_OBJECTIVE_PROBLEMS["forrester_1d"]
    results = pd.DataFrame(
        [
            {
                "problem": problem.name,
                "optimizer": "first",
                "budget": 25,
                "best_value": problem.known_minimum + 1.25,
            },
            {
                "problem": problem.name,
                "optimizer": "second",
                "budget": 25,
                "best_value": problem.known_minimum + 100.0,
            },
        ]
    )
    enriched = add_simple_regret(results, SINGLE_OBJECTIVE_PROBLEMS)
    assert enriched["regret"].tolist() == pytest.approx([1.25, 100.0])
    assert enriched["family"].tolist() == ["forrester", "forrester"]
    assert enriched["dimension"].tolist() == [1, 1]

    tiny_undershoot = results.iloc[[0]].copy()
    tiny_undershoot["best_value"] = problem.known_minimum - REGRET_TOLERANCE / 2
    assert add_simple_regret(tiny_undershoot, SINGLE_OBJECTIVE_PROBLEMS)["regret"].item() == 0.0

    bad_undershoot = results.iloc[[0]].copy()
    bad_undershoot["best_value"] = problem.known_minimum - 2 * REGRET_TOLERANCE
    with pytest.raises(ValueError, match="below registered optimum"):
        add_simple_regret(bad_undershoot, SINGLE_OBJECTIVE_PROBLEMS)


@pytest.mark.benchmarks
def test_regret_summary_keeps_budget_dimension_family_and_suite_separate() -> None:
    rows = []
    for name in ("ackley_2d", "ackley_5d", "branin_2d"):
        problem = SINGLE_OBJECTIVE_PROBLEMS[name]
        for optimizer, delta in (("left", 1.0), ("right", 2.0)):
            rows.append(
                {
                    "problem": name,
                    "optimizer": optimizer,
                    "budget": 25,
                    "status": "success",
                    "best_value": problem.known_minimum + delta,
                }
            )
    summary = summarize_regret(add_simple_regret(pd.DataFrame(rows), SINGLE_OBJECTIVE_PROBLEMS))
    assert len(summary) == 6
    assert set(summary[summary["family"] == "ackley"]["dimension"]) == {2, 5}
    assert set(summary[summary["suite"] == "synthetic"]["family"]) == {"ackley", "branin"}


@pytest.mark.benchmarks
def test_family_balanced_rank_weights_each_family_once_and_separates_suites() -> None:
    summary = pd.DataFrame(
        [
            ("synthetic", "ackley", "ackley_2d", 2, "left", 25, 0.0, 0),
            ("synthetic", "ackley", "ackley_2d", 2, "right", 25, 1.0, 0),
            ("synthetic", "ackley", "ackley_5d", 5, "left", 25, 10.0, 0),
            ("synthetic", "ackley", "ackley_5d", 5, "right", 25, 9.0, 0),
            ("synthetic", "branin", "branin_2d", 2, "left", 25, 0.0, 0),
            ("synthetic", "branin", "branin_2d", 2, "right", 25, 1.0, 0),
            ("practical", "gbr_diabetes", "gbr_diabetes", 4, "left", 25, 2.0, 0),
            ("practical", "gbr_diabetes", "gbr_diabetes", 4, "right", 25, 1.0, 0),
        ],
        columns=[
            "suite",
            "family",
            "problem",
            "dimension",
            "optimizer",
            "budget",
            "regret_median",
            "n_failed_runs",
        ],
    )
    ranks = aggregate_family_balanced_ranks(summary)
    synthetic = ranks[ranks["suite"] == "synthetic"].set_index("optimizer")
    assert synthetic.loc["left", "mean_rank"] == pytest.approx(1.25)
    assert synthetic.loc["right", "mean_rank"] == pytest.approx(1.75)
    assert synthetic["n_families"].tolist() == [2, 2]
    practical = ranks[ranks["suite"] == "practical"].set_index("optimizer")
    assert practical.loc["right", "mean_rank"] == 1.0
    assert practical.loc["left", "mean_rank"] == 2.0


@pytest.mark.benchmarks
def test_random_x2_remains_in_task_summary_but_not_primary_rank() -> None:
    problem = SINGLE_OBJECTIVE_PROBLEMS["branin_2d"]
    rows = pd.DataFrame(
        [
            {
                "problem": problem.name,
                "optimizer": "HOLA (GMM)",
                "budget": 25,
                "status": "success",
                "n_evaluations": 25,
                "best_value": problem.known_minimum + 1.0,
            },
            {
                "problem": problem.name,
                "optimizer": "TPE",
                "budget": 25,
                "status": "success",
                "n_evaluations": 25,
                "best_value": problem.known_minimum + 2.0,
            },
            {
                "problem": problem.name,
                "optimizer": "Random x2",
                "budget": 25,
                "status": "success",
                "n_evaluations": 50,
                "best_value": problem.known_minimum + 0.5,
            },
        ]
    )
    summary = summarize_regret(add_simple_regret(rows, SINGLE_OBJECTIVE_PROBLEMS))
    calibration = summary[summary["optimizer"] == "Random x2"].iloc[0]
    assert calibration["actual_evaluations_median"] == 50
    assert calibration["budget"] == 25

    ranks = aggregate_family_balanced_ranks(summary)
    assert set(ranks["optimizer"]) == {"HOLA (GMM)", "TPE"}


@pytest.mark.benchmarks
def test_multiobjective_family_balance_keeps_primary_metrics_separate() -> None:
    summary = pd.DataFrame(
        [
            ("a_3obj", "a", 3, "left", 100, 0.0, 1.0, 0),
            ("a_3obj", "a", 3, "right", 100, 1.0, 0.0, 0),
            ("a_5obj", "a", 5, "left", 100, 10.0, 0.0, 0),
            ("a_5obj", "a", 5, "right", 100, 9.0, 1.0, 0),
            ("b_2obj", "b", 2, "left", 100, 0.0, 2.0, 0),
            ("b_2obj", "b", 2, "right", 100, 1.0, 1.0, 0),
        ],
        columns=[
            "problem",
            "family",
            "n_objectives",
            "optimizer",
            "budget",
            "hv_gap_median",
            "igd_median",
            "n_failed_runs",
        ],
    )
    hv_ranks = aggregate_family_balanced_metric_ranks(summary, "hv_gap_median").set_index(
        "optimizer"
    )
    igd_ranks = aggregate_family_balanced_metric_ranks(summary, "igd_median").set_index("optimizer")
    assert hv_ranks.loc["left", "mean_rank"] == pytest.approx(1.25)
    assert hv_ranks.loc["right", "mean_rank"] == pytest.approx(1.75)
    assert igd_ranks.loc["left", "mean_rank"] == pytest.approx(1.75)
    assert igd_ranks.loc["right", "mean_rank"] == pytest.approx(1.25)
    assert hv_ranks["n_families"].tolist() == [2, 2]


@pytest.mark.benchmarks
def test_single_objective_rank_prefers_50_successes_to_one_lucky_success() -> None:
    problem = SINGLE_OBJECTIVE_PROBLEMS["forrester_1d"]
    rows = []
    for run_id in range(50):
        rows.append(
            {
                "problem": problem.name,
                "optimizer": "reliable",
                "budget": 25,
                "run_id": run_id,
                "status": "success",
                "best_value": problem.known_minimum + 10.0,
                "n_evaluations": 25,
            }
        )
        rows.append(
            {
                "problem": problem.name,
                "optimizer": "fragile",
                "budget": 25,
                "run_id": run_id,
                "status": "success" if run_id == 0 else "error",
                "best_value": problem.known_minimum if run_id == 0 else None,
                "n_evaluations": 25 if run_id == 0 else None,
            }
        )

    summary = summarize_regret(
        add_simple_regret(pd.DataFrame(rows), SINGLE_OBJECTIVE_PROBLEMS)
    ).set_index("optimizer")
    assert summary.loc["reliable", "n_total_runs"] == 50
    assert summary.loc["reliable", "n_successful_runs"] == 50
    assert summary.loc["reliable", "n_failed_runs"] == 0
    assert summary.loc["reliable", "success_rate"] == 1.0
    assert summary.loc["fragile", "n_successful_runs"] == 1
    assert summary.loc["fragile", "n_failed_runs"] == 49
    assert summary.loc["fragile", "success_rate"] == pytest.approx(0.02)

    ranks = aggregate_family_balanced_ranks(summary.reset_index()).set_index("optimizer")
    assert ranks.loc["reliable", "mean_rank"] == 1.0
    assert ranks.loc["fragile", "mean_rank"] == 2.0


@pytest.mark.benchmarks
def test_failure_rank_ties_exact_keys_and_puts_missing_metric_last_within_tier() -> None:
    index = pd.Index(["complete", "metric", "missing", "tied"])
    ranks = lexicographic_failure_ranks(
        pd.Series([0, 2, 2, 0], index=index),
        pd.Series([5.0, 1.0, float("nan"), 5.0], index=index),
    )

    assert ranks.to_dict() == {
        "complete": 1.5,
        "metric": 3.0,
        "missing": 4.0,
        "tied": 1.5,
    }


@pytest.mark.benchmarks
def test_multiobjective_rank_prefers_50_successes_to_one_lucky_success() -> None:
    problem = MULTI_OBJECTIVE_PROBLEMS["zdt1_30d"]
    rows = []
    for run_id in range(50):
        rows.append(
            {
                "problem": problem.name,
                "optimizer": "reliable",
                "budget": 100,
                "run_id": run_id,
                "status": "success",
                "normalized_hypervolume_gap": 0.5,
                "normalized_igd": 0.5,
                "spacing": 0.1,
            }
        )
        rows.append(
            {
                "problem": problem.name,
                "optimizer": "fragile",
                "budget": 100,
                "run_id": run_id,
                "status": "success" if run_id == 0 else "error",
                "normalized_hypervolume_gap": 0.01 if run_id == 0 else None,
                "normalized_igd": 0.01 if run_id == 0 else None,
                "spacing": 0.01 if run_id == 0 else None,
            }
        )

    summary = summarize_multiobjective_metrics(pd.DataFrame(rows)).set_index("optimizer")
    assert summary.loc["reliable", "n_total_runs"] == 50
    assert summary.loc["reliable", "n_successful_runs"] == 50
    assert summary.loc["reliable", "n_failed_runs"] == 0
    assert summary.loc["fragile", "n_successful_runs"] == 1
    assert summary.loc["fragile", "n_failed_runs"] == 49
    assert summary.loc["fragile", "success_rate"] == pytest.approx(0.02)

    ranks = aggregate_family_balanced_metric_ranks(
        summary.reset_index(), "hv_gap_median"
    ).set_index("optimizer")
    assert ranks.loc["reliable", "mean_rank"] == 1.0
    assert ranks.loc["fragile", "mean_rank"] == 2.0


@pytest.mark.benchmarks
def test_normalized_metrics_are_invariant_to_fixed_affine_objective_scales() -> None:
    true_front = np.array([[0.0, 1.0], [1.0, 0.0]])
    approximation = true_front[:1]
    ideal = np.array([0.0, 0.0])
    reference = np.array([2.0, 2.0])

    gap = compute_normalized_hv_gap(approximation, true_front, ideal, reference)
    igd = compute_normalized_igd(approximation, true_front, ideal, reference)
    assert gap == pytest.approx(0.25)
    assert gap == compute_normalized_hv_gap(
        approximation,
        None,
        ideal,
        reference,
        reference_hypervolume=0.75,
    )

    shifted_ideal = np.array([10.0, -5.0])
    scale = np.array([4.0, 2.0])
    shifted_true = shifted_ideal + true_front * scale
    shifted_approximation = shifted_ideal + approximation * scale
    shifted_reference = shifted_ideal + reference * scale
    assert normalize_objectives(shifted_true, shifted_ideal, shifted_reference) == pytest.approx(
        true_front / 2.0
    )
    assert compute_normalized_hv_gap(
        shifted_approximation,
        shifted_true,
        shifted_ideal,
        shifted_reference,
    ) == pytest.approx(gap)
    assert compute_normalized_igd(
        shifted_approximation,
        shifted_true,
        shifted_ideal,
        shifted_reference,
    ) == pytest.approx(igd)


@pytest.mark.benchmarks
def test_spacing_requires_three_finite_distinct_points_and_normalizes_scale() -> None:
    normalized = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [float("nan"), 0.0],
        ]
    )
    assert compute_spacing(normalized) == pytest.approx(0.0)
    assert math.isnan(compute_spacing(normalized[:2]))

    ideal = np.array([10.0, -3.0])
    reference = np.array([12.0, 7.0])
    raw = ideal + normalized * (reference - ideal)
    assert compute_normalized_spacing(raw, ideal, reference) == pytest.approx(
        compute_spacing(normalized)
    )


@pytest.mark.benchmarks
def test_registered_reference_geometry_and_analytic_dtlz_hypervolume() -> None:
    zdt3 = MULTI_OBJECTIVE_PROBLEMS["zdt3_30d"]
    assert zdt3.ideal_point is not None
    assert zdt3.ideal_point[1] == pytest.approx(-0.7733690123266403)

    zdt6 = MULTI_OBJECTIVE_PROBLEMS["zdt6_10d"]
    assert zdt6.ideal_point is not None
    assert zdt6.ideal_point[0] == pytest.approx(0.28077531881536977)

    for problem in MULTI_OBJECTIVE_PROBLEMS.values():
        assert problem.true_pareto_front is not None
        assert problem.ideal_point is not None
        normalized = normalize_objectives(
            problem.true_pareto_front,
            problem.ideal_point,
            problem.reference_point,
        )
        assert np.min(normalized) >= -1e-12
        assert np.max(normalized) < 1.0

    dtlz1 = MULTI_OBJECTIVE_PROBLEMS["dtlz1_5obj_9d"]
    expected_dtlz1_hv = 1.0 - (0.5 / 0.6) ** 5 / math.factorial(5)
    assert dtlz1.normalized_reference_hypervolume == pytest.approx(expected_dtlz1_hv)
    assert dtlz1.true_pareto_front is not None
    assert len(dtlz1.true_pareto_front) == 2048

    dtlz2 = MULTI_OBJECTIVE_PROBLEMS["dtlz2_5obj_14d"]
    orthant_ball = math.pi ** (5 / 2) / (2**5 * math.gamma(5 / 2 + 1))
    expected_dtlz2_hv = 1.0 - orthant_ball / 1.1**5
    assert dtlz2.normalized_reference_hypervolume == pytest.approx(expected_dtlz2_hv)
    assert dtlz2.true_pareto_front is not None
    assert len(dtlz2.true_pareto_front) == 2048


@pytest.mark.benchmarks
def test_registered_zdt_hypervolumes_are_continuous_front_integrals() -> None:
    reference = (1.1, 1.1)
    expected = {
        "zdt1_30d": (0.1 + 2.0 / 3.0 + 0.11) / 1.1**2,
        "zdt2_30d": (0.1 + 1.0 / 3.0 + 0.11) / 1.1**2,
        "zdt4_10d": (0.1 + 2.0 / 3.0 + 0.11) / 1.1**2,
    }
    zdt6_start = zdt.ZDT6_IDEAL_F1
    zdt6_raw = 0.1 * (1.0 - zdt6_start) + (1.0 - zdt6_start**3) / 3.0 + 0.11
    expected["zdt6_10d"] = zdt6_raw / ((1.1 - zdt6_start) * 1.1)

    for name, expected_hv in expected.items():
        problem = MULTI_OBJECTIVE_PROBLEMS[name]
        assert problem.normalized_reference_hypervolume == pytest.approx(
            expected_hv, abs=2e-15, rel=0.0
        )

    def zdt3_front(x: float) -> float:
        return 1.0 - math.sqrt(x) - x * math.sin(10.0 * math.pi * x)

    def zdt3_derivative(x: float) -> float:
        return (
            -0.5 / math.sqrt(x)
            - math.sin(10.0 * math.pi * x)
            - 10.0 * math.pi * x * math.cos(10.0 * math.pi * x)
        )

    raw_zdt3_hv = 0.0
    for index, (left, right) in enumerate(zdt.ZDT3_PARETO_INTERVALS):
        raw_zdt3_hv += quad(
            lambda x: reference[1] - zdt3_front(x),
            left,
            right,
            epsabs=1e-14,
            epsrel=1e-14,
        )[0]
        next_left = (
            zdt.ZDT3_PARETO_INTERVALS[index + 1][0]
            if index + 1 < len(zdt.ZDT3_PARETO_INTERVALS)
            else reference[0]
        )
        raw_zdt3_hv += (next_left - right) * (reference[1] - zdt3_front(right))
        assert zdt3_derivative(right) == pytest.approx(0.0, abs=2e-13, rel=0.0)
        if index:
            previous_right = zdt.ZDT3_PARETO_INTERVALS[index - 1][1]
            assert zdt3_front(left) == pytest.approx(zdt3_front(previous_right), abs=2e-15, rel=0.0)

    expected_zdt3_hv = raw_zdt3_hv / (1.1 * (1.1 - zdt.ZDT3_IDEAL_F2))
    zdt3_problem = MULTI_OBJECTIVE_PROBLEMS["zdt3_30d"]
    assert zdt3_problem.normalized_reference_hypervolume == pytest.approx(
        expected_zdt3_hv, abs=2e-14, rel=0.0
    )
    assert expected_zdt3_hv == pytest.approx(0.6462653889107179, abs=2e-14, rel=0.0)


@pytest.mark.benchmarks
def test_sampled_zdt_fronts_have_positive_gap_to_analytic_continuous_fronts() -> None:
    for problem in MULTI_OBJECTIVE_PROBLEMS.values():
        if not problem.family.startswith("zdt"):
            continue
        assert problem.true_pareto_front is not None
        assert problem.ideal_point is not None
        sampled_gap = compute_normalized_hv_gap(
            problem.true_pareto_front,
            None,
            problem.ideal_point,
            problem.reference_point,
            reference_hypervolume=problem.normalized_reference_hypervolume,
        )
        assert sampled_gap > 0.0


@pytest.mark.benchmarks
def test_reference_geometry_rejects_true_front_on_reporting_boundary() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        MultiObjectiveProblem(
            name="invalid",
            func=lambda params: {"f1": params["x"], "f2": 1.0 - params["x"]},
            bounds={"x": (0.0, 1.0)},
            objective_names=("f1", "f2"),
            ideal_point=(0.0, 0.0),
            reference_point=(1.0, 1.0),
            true_pareto_front=np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
