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
Test suite for the HOLA Python API.

Tests Study, Space, Trial, CompletedTrial, and parameter/objective types:
parameter attributes, space construction, ask/tell lifecycle, strategy
variants, multi-param spaces, objectives, Trial repr, Study.connect(),
categorical parameters, and study.run().
"""

import pytest

# ==========================================================================
# 1. Parameter Types & Spaces
# ==========================================================================


def test_continuous_attributes():
    from hola import Real

    c = Real(min=0.0, max=1.0)
    assert c.min == 0.0
    assert c.max == 1.0


def test_log10_attributes():
    """Log10 stores actual values (not exponents)."""
    from hola import Real

    log_p = Real(min=1e-4, max=0.1, scale="log10")
    assert abs(log_p.min - 1e-4) < 1e-12
    assert abs(log_p.max - 0.1) < 1e-12


def test_discrete_attributes():
    from hola import Integer

    d = Integer(min=1, max=10)
    assert d.min == 1
    assert d.max == 10


def test_minimize_attributes():
    from hola import Minimize

    m = Minimize(field="loss", priority=1.0)
    assert m.field == "loss"
    assert m.priority == 1.0
    assert m.target is None
    assert m.limit is None


def test_maximize_attributes():
    from hola import Maximize

    m = Maximize(field="acc", target=0.95, limit=0.5, priority=2.0)
    assert m.field == "acc"
    assert m.target == 0.95
    assert m.limit == 0.5
    assert m.priority == 2.0


def test_space_builder():
    from hola import Integer, Real, Space

    space = Space(lr=Real(1e-4, 0.1, scale="log10"), layers=Integer(1, 10), weight=Real(0.0, 1.0))
    assert space is not None


def test_space_bad_param_type():
    from hola import Space

    with pytest.raises(ValueError):
        Space(x="not a param type")  # type: ignore[arg-type]


# ==========================================================================
# 2. Local Study Lifecycle
# ==========================================================================


def test_study_creation():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    assert study.trial_count() == 0
    assert len(study.top_k(1)) == 0


def test_study_ask_returns_trial():
    from hola import Minimize, Real, Space, Study, Trial

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t = study.ask()
    assert isinstance(t, Trial)
    assert t.trial_id == 0
    assert isinstance(t.params, dict)
    assert "x" in t.params


def test_study_monotonic_ids():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t0 = study.ask()
    t1 = study.ask()
    t2 = study.ask()
    assert t0.trial_id == 0
    assert t1.trial_id == 1
    assert t2.trial_id == 2


def test_study_params_in_bounds():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    for _ in range(20):
        t = study.ask()
        assert 0.0 <= t.params["x"] <= 1.0
        study.tell(t.trial_id, {"loss": t.params["x"]})


def test_study_tell_increments_count():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t0 = study.ask()
    t1 = study.ask()
    study.tell(t0.trial_id, {"loss": 0.8})
    assert study.trial_count() == 1
    study.tell(t1.trial_id, {"loss": 0.2})
    assert study.trial_count() == 2


def test_study_best():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t0 = study.ask()
    t1 = study.ask()
    study.tell(t0.trial_id, {"loss": 0.8})
    study.tell(t1.trial_id, {"loss": 0.2})

    top = study.top_k(1)
    assert len(top) == 1
    best = top[0]
    assert best.trial_id == t1.trial_id
    assert list(best.score_vector.values())[0] <= 0.2 + 1e-9
    assert isinstance(best.params, dict)


def test_study_unknown_trial_raises():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    with pytest.raises(ValueError):
        study.tell(999, {"loss": 0.5})


# ==========================================================================
# 3. Strategy Variants
# ==========================================================================


def test_study_strategy_random():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")], strategy="random")
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_study_strategy_sobol():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")], strategy="sobol")
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_study_strategy_gmm():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")], strategy="gmm")
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


# ==========================================================================
# 4. Multi-Param & Objectives
# ==========================================================================


def test_study_multi_param():
    from hola import Integer, Minimize, Real, Space, Study

    study = Study(
        space=Space(
            lr=Real(1e-4, 0.1, scale="log10"),
            layers=Integer(1, 10),
            weight=Real(0.0, 1.0),
        ),
        objectives=[Minimize("loss")],
        strategy="sobol",
    )
    t = study.ask()
    assert "lr" in t.params
    assert "layers" in t.params
    assert "weight" in t.params
    # Log10 param is in actual domain (not log space)
    assert 1e-4 <= t.params["lr"] <= 0.1 + 1e-9
    assert 1 <= t.params["layers"] <= 10
    assert 0.0 <= t.params["weight"] <= 1.0


def test_study_with_objectives():
    from hola import Maximize, Minimize, Real, Space, Study

    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss"), Maximize("accuracy")],
    )
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.3, "accuracy": 0.9})
    assert study.trial_count() == 1


# ==========================================================================
# 5. Trial repr
# ==========================================================================


def test_trial_repr():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t = study.ask()
    r = repr(t)
    assert "Trial" in r
    assert "trial_id=0" in r


# ==========================================================================
# 6. Study.connect() (replacement for RemoteStudy)
# ==========================================================================


def test_study_connect_returns_study():
    from hola import Study

    # Study.connect() returns a Study object (connection is lazy)
    remote = Study.connect("http://localhost:9999")
    assert remote is not None
    # Actual connection errors happen on first ask/tell
    with pytest.raises(ValueError):
        remote.ask()


# ==========================================================================
# 7. Categorical Parameters
# ==========================================================================


def test_categorical_attributes():
    from hola import Categorical

    c = Categorical(choices=["adam", "sgd", "rmsprop"])
    assert c.choices == ["adam", "sgd", "rmsprop"]


def test_space_with_categorical():
    from hola import Categorical, Real, Space

    space = Space(opt=Categorical(["adam", "sgd"]), lr=Real(0.0, 1.0))
    assert space is not None


def test_study_categorical_ask_tell():
    from hola import Categorical, Minimize, Space, Study

    study = Study(
        space=Space(opt=Categorical(["adam", "sgd", "rmsprop"])), objectives=[Minimize("loss")]
    )
    t = study.ask()
    assert isinstance(t.params["opt"], str)
    assert t.params["opt"] in ["adam", "sgd", "rmsprop"]
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_study_categorical_all_in_bounds():
    from hola import Categorical, Minimize, Space, Study

    study = Study(space=Space(algo=Categorical(["a", "b", "c"])), objectives=[Minimize("loss")])
    for _ in range(20):
        t = study.ask()
        assert t.params["algo"] in ["a", "b", "c"]
        study.tell(t.trial_id, {"loss": 0.5})


def test_study_categorical_mixed_space():
    from hola import Categorical, Integer, Minimize, Real, Space, Study

    study = Study(
        space=Space(
            x=Real(0.0, 1.0),
            n=Integer(1, 5),
            opt=Categorical(["adam", "sgd"]),
        ),
        objectives=[Minimize("loss")],
    )
    t = study.ask()
    assert 0.0 <= t.params["x"] <= 1.0
    assert 1 <= t.params["n"] <= 5
    assert t.params["opt"] in ["adam", "sgd"]
    study.tell(t.trial_id, {"loss": 0.42})
    assert study.trial_count() == 1


# ==========================================================================
# 8. Study.run() Convenience Method
# ==========================================================================


def test_study_run_basic():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    def objective(params):
        return {"loss": params["x"] ** 2}

    study.run(objective, n_trials=10)
    assert study.trial_count() == 10


def test_study_run_returns_self():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    result = study.run(lambda p: {"loss": p["x"]}, n_trials=5)
    # Should return the same study object (for chaining)
    assert result is study


def test_study_run_chain_best():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    top = study.run(lambda p: {"loss": p["x"] ** 2}, n_trials=20).top_k(1)
    assert len(top) == 1
    best = top[0]
    assert isinstance(best.score_vector, dict)
    assert isinstance(best.params, dict)


def test_study_run_bad_return_raises():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    with pytest.raises(ValueError, match="dict"):
        study.run(lambda p: 42, n_trials=1)  # Returns int, not dict


def test_study_run_parallel():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    study.run(lambda p: {"loss": p["x"] ** 2}, n_trials=20, n_workers=4)
    assert study.trial_count() == 20
    top = study.top_k(1)
    assert len(top) == 1
    assert list(top[0].score_vector.values())[0] < 1.0


def test_study_run_parallel_default_workers():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    # n_workers=1 (default) runs sequentially
    study.run(lambda p: {"loss": p["x"]}, n_trials=10)
    assert study.trial_count() == 10


def test_study_run_sequential_explicit():
    from hola import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    study.run(lambda p: {"loss": p["x"]}, n_trials=10, n_workers=1)
    assert study.trial_count() == 10


# ==========================================================================
# Seed determinism
# ==========================================================================


def test_seed_determinism():
    from hola import Minimize, Real, Space, Study

    def run_with_seed(seed):
        study = Study(
            space=Space(x=Real(0.0, 1.0)),
            objectives=[Minimize("loss")],
            strategy="sobol",
            seed=seed,
        )
        results = []
        for _ in range(5):
            t = study.ask()
            results.append(t.params["x"])
        return results

    r1 = run_with_seed(42)
    r2 = run_with_seed(42)
    assert r1 == r2, "Same seed should produce identical candidates"


def test_different_seeds_differ():
    from hola import Minimize, Real, Space, Study

    def first_param(seed):
        study = Study(
            space=Space(x=Real(0.0, 1.0)),
            objectives=[Minimize("loss")],
            strategy="random",
            seed=seed,
        )
        return study.ask().params["x"]

    assert first_param(1) != first_param(2), "Different seeds should differ"


# ==========================================================================
# Pareto front
# ==========================================================================


def test_pareto_front_multi_objective():
    from hola import Minimize, Real, Space, Study

    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[
            Minimize("f1", target=0.0, limit=10.0, priority=1.0),
            Minimize("f2", target=0.0, limit=10.0, priority=2.0),
        ],
        seed=0,
    )

    study.run(
        lambda p: {"f1": p["x"], "f2": 1.0 - p["x"]},
        n_trials=20,
        n_workers=1,
    )

    front = study.pareto_front()
    assert len(front) > 0
    for trial in front:
        assert hasattr(trial, "trial_id")
        assert isinstance(trial.params, dict)
        assert isinstance(trial.scores, dict)
        assert "f1" in trial.scores
        assert "f2" in trial.scores


def test_pareto_front_scalar_returns_empty():
    from hola import Minimize, Real, Space, Study

    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
    )
    # Scalar (single-objective) study returns empty list instead of raising
    front = study.pareto_front()
    assert front == []


# ==========================================================================
# Trials
# ==========================================================================


def test_trials_returns_all():
    from hola import Minimize, Real, Space, Study

    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
        seed=0,
    )
    study.run(lambda p: {"loss": p["x"] ** 2}, n_trials=15, n_workers=1)

    all_trials = study.trials()
    assert len(all_trials) == 15
    for trial in all_trials:
        assert hasattr(trial, "trial_id")
        assert isinstance(trial.params, dict)
        assert isinstance(trial.score_vector, dict)
