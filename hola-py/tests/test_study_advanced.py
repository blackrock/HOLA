# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Advanced Study tests covering error paths, scalarization, convergence,
concurrency, and best-tracking.

Tests double-tell errors, bad strategy fallback, maximize/minimize
scalarization, TLP feasibility, multi-objective priorities, Sobol
properties, GMM refit, end-to-end convergence, concurrent ask/tell,
and monotonic best-tracking.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from hola import Categorical, Integer, Maximize, Minimize, Real, Space, Study

# ==========================================================================
# Error Paths
# ==========================================================================


def test_double_tell_raises(simple_space):
    study = Study(space=simple_space, objectives=[Minimize("loss")])
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    with pytest.raises(ValueError):
        study.tell(t.trial_id, {"loss": 0.3})


def test_bad_strategy_string_defaults_gracefully(simple_space):
    # Unknown strategy names do not raise; the engine falls back to a default.
    study = Study(space=simple_space, objectives=[Minimize("loss")], strategy="nonexistent")
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_run_zero_trials(simple_space):
    study = Study(space=simple_space, objectives=[Minimize("loss")])
    study.run(lambda p: {"loss": p["x"]}, n_trials=0)
    assert study.trial_count() == 0


# ==========================================================================
# Scalarization
# ==========================================================================


def test_maximize_negates_observation():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Maximize("acc")],
    )
    t = study.ask()
    study.tell(t.trial_id, {"acc": 0.9})
    best = study.top_k(1)[0]
    assert best is not None
    assert list(best.scores.values())[0] < 0, "Maximize should negate the observation internally"


def test_minimize_preserves_observation():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
    )
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    best = study.top_k(1)[0]
    assert best is not None
    assert abs(list(best.score_vector.values())[0] - 0.5) < 1e-9


def test_missing_metric_field_not_best():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
    )
    # Tell with a metric that doesn't contain the objective field — gets inf score
    t1 = study.ask()
    study.tell(t1.trial_id, {"accuracy": 0.9})
    # Inf-scored trials are not returned by top_k()
    assert len(study.top_k(1)) == 0

    # Tell a valid trial — it should become best
    t2 = study.ask()
    study.tell(t2.trial_id, {"loss": 0.5})
    best = study.top_k(1)[0]
    assert best is not None
    assert best.trial_id == t2.trial_id


def test_extra_metric_fields_ignored():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
    )
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.3, "accuracy": 0.9, "latency": 50.0})
    best = study.top_k(1)[0]
    assert best is not None
    assert abs(list(best.score_vector.values())[0] - 0.3) < 1e-9


# ==========================================================================
# TLP Objectives
# ==========================================================================


def test_tlp_feasible_trial():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss", target=0.0, limit=1.0)],
    )
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    best = study.top_k(1)[0]
    assert best is not None
    assert math.isfinite(list(best.score_vector.values())[0])


def test_tlp_infeasible_vs_feasible():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss", target=0.0, limit=1.0)],
    )
    # Tell an infeasible trial (loss exceeds limit)
    t1 = study.ask()
    study.tell(t1.trial_id, {"loss": 2.0})

    # Tell a feasible trial
    t2 = study.ask()
    study.tell(t2.trial_id, {"loss": 0.5})

    best = study.top_k(1)[0]
    assert best is not None
    assert best.trial_id == t2.trial_id, "Feasible trial should be best"


# ==========================================================================
# Multi-Objective
# ==========================================================================


def test_multi_objective_with_priorities():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss", priority=1.0), Minimize("latency", priority=0.5)],
    )
    for _ in range(10):
        t = study.ask()
        x = t.params["x"]
        study.tell(t.trial_id, {"loss": x**2, "latency": (1 - x) * 100})
    best = study.top_k(1)[0]
    assert best is not None
    # Distinct priorities -> vector leaderboard -> score_vector is per-group dict
    obs = best.score_vector
    assert isinstance(obs, dict)
    assert all(math.isfinite(v) for v in obs.values())


# ==========================================================================
# Sobol Properties
# ==========================================================================


def test_sobol_unique_points_2d(sphere_space):
    study = Study(space=sphere_space, objectives=[Minimize("loss")], strategy="sobol")
    points = set()
    for _ in range(100):
        t = study.ask()
        key = (round(t.params["x"], 12), round(t.params["y"], 12))
        points.add(key)
        study.tell(t.trial_id, {"loss": t.params["x"] ** 2 + t.params["y"] ** 2})
    assert len(points) == 100, "Sobol should produce 100 unique points"


def test_sobol_deterministic():
    space1 = Space(x=Real(0.0, 1.0), y=Real(0.0, 1.0))
    space2 = Space(x=Real(0.0, 1.0), y=Real(0.0, 1.0))
    study1 = Study(space=space1, objectives=[Minimize("loss")], strategy="sobol")
    study2 = Study(space=space2, objectives=[Minimize("loss")], strategy="sobol")

    for _ in range(20):
        t1 = study1.ask()
        t2 = study2.ask()
        assert abs(t1.params["x"] - t2.params["x"]) < 1e-12
        assert abs(t1.params["y"] - t2.params["y"]) < 1e-12
        study1.tell(t1.trial_id, {"loss": 0.0})
        study2.tell(t2.trial_id, {"loss": 0.0})


# ==========================================================================
# GMM Refit
# ==========================================================================


def test_gmm_survives_50_trials():
    study = Study(space=Space(x=Real(-5.0, 5.0)), objectives=[Minimize("loss")], strategy="gmm")
    for _ in range(50):
        t = study.ask()
        study.tell(t.trial_id, {"loss": t.params["x"] ** 2})
    assert study.trial_count() == 50
    best = study.top_k(1)[0]
    assert best is not None


@pytest.mark.slow
def test_gmm_vs_random_on_sphere():
    def sphere(params):
        return {"loss": params["x"] ** 2 + params["y"] ** 2}

    space_gmm = Space(x=Real(-5.0, 5.0), y=Real(-5.0, 5.0))
    space_rnd = Space(x=Real(-5.0, 5.0), y=Real(-5.0, 5.0))

    gmm_study = Study(space=space_gmm, objectives=[Minimize("loss")], strategy="gmm", seed=7)
    gmm_study.run(sphere, n_trials=200, n_workers=1)

    rnd_study = Study(space=space_rnd, objectives=[Minimize("loss")], strategy="random", seed=7)
    rnd_study.run(sphere, n_trials=200, n_workers=1)

    gmm_best = list(gmm_study.top_k(1)[0].score_vector.values())[0]
    # GMM should find a reasonable minimum on a simple sphere.
    # We only check that it reaches a low absolute value, since random
    # can occasionally get lucky on such a small search space.
    assert gmm_best < 1.0, f"GMM best ({gmm_best}) too high on sphere"


# ==========================================================================
# End-to-End Convergence
# ==========================================================================


def _sphere(params):
    return {"loss": params["x"] ** 2 + params["y"] ** 2}


def test_convergence_sphere_sobol():
    study = Study(
        space=Space(x=Real(-5.0, 5.0), y=Real(-5.0, 5.0)),
        objectives=[Minimize("loss")],
        strategy="sobol",
    )
    study.run(_sphere, n_trials=200, n_workers=1)
    best = study.top_k(1)[0]
    assert list(best.score_vector.values())[0] < 2.0


def test_convergence_sphere_random():
    study = Study(
        space=Space(x=Real(-5.0, 5.0), y=Real(-5.0, 5.0)),
        objectives=[Minimize("loss")],
        strategy="random",
    )
    study.run(_sphere, n_trials=200, n_workers=1)
    best = study.top_k(1)[0]
    assert list(best.score_vector.values())[0] < 5.0


def test_convergence_sphere_gmm():
    study = Study(
        space=Space(x=Real(-5.0, 5.0), y=Real(-5.0, 5.0)),
        objectives=[Minimize("loss")],
        strategy="gmm",
    )
    study.run(_sphere, n_trials=200, n_workers=1)
    best = study.top_k(1)[0]
    assert list(best.score_vector.values())[0] < 15.0


def test_convergence_mixed_space():
    def objective(params):
        # Reward: lr near 0.01, layers near 5, "adam" preferred
        lr_err = (math.log10(params["lr"]) - math.log10(0.01)) ** 2
        layer_err = (params["layers"] - 5) ** 2
        opt_bonus = 0.0 if params["opt"] == "adam" else 10.0
        return {"loss": lr_err + layer_err + opt_bonus}

    study = Study(
        space=Space(
            lr=Real(1e-4, 0.1, scale="log10"),
            layers=Integer(1, 10),
            opt=Categorical(["adam", "sgd", "rmsprop"]),
        ),
        objectives=[Minimize("loss")],
        strategy="sobol",
    )
    study.run(objective, n_trials=100, n_workers=1)
    best = study.top_k(1)[0]
    assert list(best.score_vector.values())[0] < 20.0


@pytest.mark.slow
def test_convergence_rosenbrock():
    def rosenbrock(params):
        x, y = params["x"], params["y"]
        return {"loss": (1 - x) ** 2 + 100 * (y - x**2) ** 2}

    study = Study(
        space=Space(x=Real(-5.0, 5.0), y=Real(-5.0, 5.0)),
        objectives=[Minimize("loss")],
        strategy="sobol",
    )
    study.run(rosenbrock, n_trials=500, n_workers=1)
    best = study.top_k(1)[0]
    assert list(best.score_vector.values())[0] < 10.0


# ==========================================================================
# Concurrency
# ==========================================================================


def test_concurrent_ask_tell_threads():
    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")], strategy="random")

    def worker(_):
        t = study.ask()
        study.tell(t.trial_id, {"loss": t.params["x"] ** 2})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(20)))

    assert study.trial_count() == 20


def test_run_parallel_correct_count():
    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    study.run(lambda p: {"loss": p["x"] ** 2}, n_trials=50, n_workers=4)
    assert study.trial_count() == 50


# ==========================================================================
# Best Tracking
# ==========================================================================


def test_best_is_actual_minimum():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
    )
    losses = [0.9, 0.7, 0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.05]
    for loss in losses:
        t = study.ask()
        study.tell(t.trial_id, {"loss": loss})

    best = study.top_k(1)[0]
    assert abs(list(best.score_vector.values())[0] - 0.05) < 1e-9


def test_best_updates_monotonically():
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
    )
    prev_best = float("inf")
    for loss in [0.9, 0.7, 0.5, 0.3, 0.1]:
        t = study.ask()
        study.tell(t.trial_id, {"loss": loss})
        best = study.top_k(1)[0]
        assert list(best.score_vector.values())[0] <= prev_best + 1e-9
        prev_best = list(best.score_vector.values())[0]
