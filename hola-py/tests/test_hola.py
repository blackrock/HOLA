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
    from hola_opt import Real

    c = Real(min=0.0, max=1.0)
    assert c.min == 0.0
    assert c.max == 1.0


def test_log10_attributes():
    """Log10 stores actual values (not exponents)."""
    from hola_opt import Real

    log_p = Real(min=1e-4, max=0.1, scale="log10")
    assert abs(log_p.min - 1e-4) < 1e-12
    assert abs(log_p.max - 0.1) < 1e-12


def test_discrete_attributes():
    from hola_opt import Integer

    d = Integer(min=1, max=10)
    assert d.min == 1
    assert d.max == 10


def test_minimize_attributes():
    from hola_opt import Minimize

    m = Minimize(field="loss", priority=1.0)
    assert m.field == "loss"
    assert m.priority == 1.0
    assert m.target is None
    assert m.limit is None


def test_maximize_attributes():
    from hola_opt import Maximize

    m = Maximize(field="acc", target=0.95, limit=0.5, priority=2.0)
    assert m.field == "acc"
    assert m.target == 0.95
    assert m.limit == 0.5
    assert m.priority == 2.0


def test_space_builder():
    from hola_opt import Integer, Real, Space

    space = Space(lr=Real(1e-4, 0.1, scale="log10"), layers=Integer(1, 10), weight=Real(0.0, 1.0))
    assert space is not None


def test_space_bad_param_type():
    from hola_opt import Space

    with pytest.raises(ValueError):
        Space(x="not a param type")  # type: ignore[arg-type]


# ==========================================================================
# 2. Local Study Lifecycle
# ==========================================================================


def test_study_creation():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    assert study.trial_count() == 0
    assert len(study.top_k(1)) == 0


def test_study_ask_returns_trial():
    from hola_opt import Minimize, Real, Space, Study, Trial

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t = study.ask()
    assert isinstance(t, Trial)
    assert t.trial_id == 0
    assert isinstance(t.params, dict)
    assert "x" in t.params


def test_study_monotonic_ids():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t0 = study.ask()
    t1 = study.ask()
    t2 = study.ask()
    assert t0.trial_id == 0
    assert t1.trial_id == 1
    assert t2.trial_id == 2


def test_study_params_in_bounds():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    for _ in range(20):
        t = study.ask()
        assert 0.0 <= t.params["x"] <= 1.0
        study.tell(t.trial_id, {"loss": t.params["x"]})


def test_study_tell_increments_count():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t0 = study.ask()
    t1 = study.ask()
    study.tell(t0.trial_id, {"loss": 0.8})
    assert study.trial_count() == 1
    study.tell(t1.trial_id, {"loss": 0.2})
    assert study.trial_count() == 2


def test_study_best():
    from hola_opt import Minimize, Real, Space, Study

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
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    with pytest.raises(ValueError):
        study.tell(999, {"loss": 0.5})


# ==========================================================================
# 3. Strategy Variants
# ==========================================================================


def test_study_strategy_random():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")], strategy="random")
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_study_strategy_sobol():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")], strategy="sobol")
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_study_strategy_gmm():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")], strategy="gmm")
    t = study.ask()
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


# ==========================================================================
# 4. Multi-Param & Objectives
# ==========================================================================


def test_study_multi_param():
    from hola_opt import Integer, Minimize, Real, Space, Study

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
    from hola_opt import Maximize, Minimize, Real, Space, Study

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
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t = study.ask()
    r = repr(t)
    assert "Trial" in r
    assert "trial_id=0" in r


# ==========================================================================
# 6. Study.connect() (replacement for RemoteStudy)
# ==========================================================================


def test_study_connect_returns_study():
    from hola_opt import Study

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
    from hola_opt import Categorical

    c = Categorical(choices=["adam", "sgd", "rmsprop"])
    assert c.choices == ["adam", "sgd", "rmsprop"]


def test_space_with_categorical():
    from hola_opt import Categorical, Real, Space

    space = Space(opt=Categorical(["adam", "sgd"]), lr=Real(0.0, 1.0))
    assert space is not None


def test_study_categorical_ask_tell():
    from hola_opt import Categorical, Minimize, Space, Study

    study = Study(
        space=Space(opt=Categorical(["adam", "sgd", "rmsprop"])), objectives=[Minimize("loss")]
    )
    t = study.ask()
    assert isinstance(t.params["opt"], str)
    assert t.params["opt"] in ["adam", "sgd", "rmsprop"]
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_study_categorical_all_in_bounds():
    from hola_opt import Categorical, Minimize, Space, Study

    study = Study(space=Space(algo=Categorical(["a", "b", "c"])), objectives=[Minimize("loss")])
    for _ in range(20):
        t = study.ask()
        assert t.params["algo"] in ["a", "b", "c"]
        study.tell(t.trial_id, {"loss": 0.5})


def test_study_categorical_mixed_space():
    from hola_opt import Categorical, Integer, Minimize, Real, Space, Study

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
# 8. Checkpoint Persistence
# ==========================================================================


def test_study_save_load_resume_uses_fresh_trial_id(tmp_path):
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    for expected_id, loss in enumerate([0.5, 0.3]):
        trial = study.ask()
        assert trial.trial_id == expected_id
        completed = study.tell(trial.trial_id, {"loss": loss})
        assert completed.trial_id == expected_id

    path = tmp_path / "study.json"
    study.save(str(path))

    restored = Study.load(str(path))
    trial = restored.ask()
    assert trial.trial_id == 2
    completed = restored.tell(trial.trial_id, {"loss": 0.1})
    assert completed.trial_id == 2
    assert completed.params == trial.params
    assert [trial.trial_id for trial in restored.trials()] == [0, 1, 2]


def test_study_save_load_resume_uses_fresh_vector_trial_id(tmp_path):
    from hola_opt import Minimize, Real, Space, Study

    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("f1", priority=1.0), Minimize("f2", priority=2.0)],
    )
    for expected_id, metrics in enumerate([{"f1": 1.0, "f2": 3.0}, {"f1": 2.0, "f2": 1.0}]):
        trial = study.ask()
        assert trial.trial_id == expected_id
        completed = study.tell(trial.trial_id, metrics)
        assert completed.trial_id == expected_id

    path = tmp_path / "vector-study.json"
    study.save(str(path))

    restored = Study.load(str(path))
    trial = restored.ask()
    assert trial.trial_id == 2
    completed = restored.tell(trial.trial_id, {"f1": 0.5, "f2": 2.5})
    assert completed.trial_id == 2
    assert [trial.trial_id for trial in restored.trials()] == [0, 1, 2]


# ==========================================================================
# 9. Study.run() Convenience Method
# ==========================================================================


def test_study_run_basic():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    def objective(params):
        return {"loss": params["x"] ** 2}

    study.run(objective, n_trials=10)
    assert study.trial_count() == 10


def test_study_run_returns_self():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    result = study.run(lambda p: {"loss": p["x"]}, n_trials=5)
    # Should return the same study object (for chaining)
    assert result is study


def test_study_run_chain_best():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    top = study.run(lambda p: {"loss": p["x"] ** 2}, n_trials=20).top_k(1)
    assert len(top) == 1
    best = top[0]
    assert isinstance(best.score_vector, dict)
    assert isinstance(best.params, dict)


def test_study_run_bad_return_raises():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    with pytest.raises(ValueError, match="dict"):
        study.run(lambda p: 42, n_trials=1)  # Returns int, not dict


def test_study_run_parallel():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    study.run(lambda p: {"loss": p["x"] ** 2}, n_trials=20, n_workers=4)
    assert study.trial_count() == 20
    top = study.top_k(1)
    assert len(top) == 1
    assert list(top[0].score_vector.values())[0] < 1.0


def test_study_run_parallel_integrity():
    """Parallel run() must evaluate every trial exactly once, with no drops,
    duplicates, or mismatched score<->params pairings.

    The existing parallel smoke test only checks trial_count()==n and best<1.0,
    which would still pass if the executor silently dropped some evaluations and
    re-ran others, or if a result were recorded against the wrong trial's params.
    Here the objective records each invocation's params (under a lock, since it
    runs on several worker threads) and we cross-check the engine's record of
    completed trials against those invocations:

      * the objective is invoked EXACTLY n_trials times (no drops/duplicates);
      * the completed trial ids are distinct and number exactly n_trials;
      * every completed trial's stored score equals the objective recomputed
        from that same trial's params (no params/score crossing).
    """
    import threading

    from hola_opt import Minimize, Real, Space, Study

    n_trials = 20
    n_workers = 4

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    lock = threading.Lock()
    recorded_calls: list[dict] = []

    def score_of(x: float) -> float:
        # Deterministic function of the params alone, so a completed trial's
        # score can be recomputed solely from its recorded params.
        return x**2

    def objective(params):
        with lock:
            recorded_calls.append(dict(params))
        return {"loss": score_of(params["x"])}

    study.run(objective, n_trials=n_trials, n_workers=n_workers)

    # The objective fired exactly once per trial: no dropped or duplicated work.
    assert len(recorded_calls) == n_trials

    completed = study.trials()
    assert len(completed) == n_trials

    # Completed trial ids are distinct and exactly n_trials of them.
    ids = [t.trial_id for t in completed]
    assert len(set(ids)) == n_trials

    # Every stored score matches the objective recomputed from that trial's own
    # params, proving no params/score pair was crossed between trials.
    for t in completed:
        expected = score_of(t.params["x"])
        actual = list(t.score_vector.values())[0]
        assert actual == pytest.approx(expected, rel=0, abs=1e-12), (
            f"trial {t.trial_id}: stored score {actual} != "
            f"recomputed {expected} from params {t.params}"
        )

    # The multiset of x-values the objective saw matches the multiset stored on
    # the completed trials: every evaluation landed on exactly one trial and no
    # trial carries a value the objective never produced.
    seen_x = sorted(c["x"] for c in recorded_calls)
    stored_x = sorted(t.params["x"] for t in completed)
    assert seen_x == pytest.approx(stored_x, rel=0, abs=0.0)


def test_study_run_parallel_default_workers():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    # n_workers=1 (default) runs sequentially
    study.run(lambda p: {"loss": p["x"]}, n_trials=10)
    assert study.trial_count() == 10


def test_study_run_sequential_explicit():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    study.run(lambda p: {"loss": p["x"]}, n_trials=10, n_workers=1)
    assert study.trial_count() == 10


# ==========================================================================
# 9b. Study.run() orphan cancellation on objective failure
# ==========================================================================


def test_study_run_failure_cancels_orphan_trial_sequential():
    """Sequential run() that fails midway must cancel its pending trial.

    A trial is asked for before the objective runs; if the objective raises,
    run() must cancel that trial so it does not linger in the engine's pending
    set. We verify both that the failed trial_id is no longer tellable (it was
    cancelled, not orphaned) and that the engine recovers cleanly.
    """
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    calls = {"n": 0}

    def flaky_objective(params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"loss": params["x"]}

    with pytest.raises(RuntimeError, match="boom"):
        study.run(flaky_objective, n_trials=5)

    # First trial completed; the second was asked then cancelled on failure.
    assert study.trial_count() == 1

    # Trials are asked with monotonic ids in the sequential path, so the failed
    # trial's id is the completed count at the moment it was asked. It was
    # cancelled, so telling it now must raise rather than succeed.
    failed_trial_id = 1
    with pytest.raises(ValueError):
        study.tell(failed_trial_id, {"loss": 0.5})

    # Engine is not wedged: a fresh successful run completes fully.
    study.run(lambda p: {"loss": p["x"]}, n_trials=3)
    assert study.trial_count() == 4


def test_study_run_failure_cancels_orphan_trials_parallel():
    """Parallel run() that fails midway must cancel outstanding trials.

    With multiple workers a whole batch of trials is asked for (and thus
    pending in the engine) before any result is told. If the objective raises,
    run() must cancel every still-outstanding trial; otherwise they linger in
    the engine's pending set and keep consuming the exploration budget, since
    ask() gates new trials on completed + pending against max_trials.

    Outstanding ids are not individually observable, so we make the leak
    observable through the budget. The study is created with a tight
    max_trials equal to the worker batch size, so the failing run asks the
    entire budget at once. With the orphans cancelled the freed budget lets a
    subsequent clean run complete; if the cancel loop is reverted the leaked
    pending trials keep the budget exhausted and the clean run's first ask
    fails. trial_count() (completed only) is identical in both cases, so the
    discriminating signal is whether the clean run completes at all.
    """
    from hola_opt import Minimize, Real, Space, Study

    workers = 4
    study = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
        max_trials=workers,
    )

    calls = {"n": 0}

    def flaky_objective(params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"loss": params["x"]}

    # The first batch asks for all `workers` trials at once (the whole budget),
    # then one of them raises while collecting results.
    with pytest.raises(RuntimeError, match="boom"):
        study.run(flaky_objective, n_trials=8, n_workers=workers)

    count_after_failure = study.trial_count()
    # At least one trial in the batch raised and was never told, so the budget
    # is not fully accounted for by completed trials.
    assert count_after_failure < workers

    # Budget freed by cancelling the orphaned pending trials. If they were left
    # pending instead, the budget would stay at `workers` and the clean run
    # below could not ask a single trial.
    remaining = workers - count_after_failure

    # The freed budget lets this run complete fully; leaked pending trials would
    # keep max_trials exhausted and the first ask() of this run would raise.
    study.run(lambda p: {"loss": p["x"]}, n_trials=remaining, n_workers=workers)
    assert study.trial_count() == count_after_failure + remaining
    assert study.trial_count() == workers


def test_study_run_bad_return_cancels_orphan_trial():
    """A non-dict objective return must also cancel the pending trial."""
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    with pytest.raises(ValueError, match="dict"):
        study.run(lambda p: 42, n_trials=1)

    # The single asked trial was cancelled, not left pending.
    with pytest.raises(ValueError):
        study.tell(0, {"loss": 0.5})

    # Engine recovers cleanly.
    study.run(lambda p: {"loss": p["x"]}, n_trials=3)
    assert study.trial_count() == 3


# ==========================================================================
# Seed determinism
# ==========================================================================


def test_seed_determinism():
    from hola_opt import Minimize, Real, Space, Study

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
    from hola_opt import Minimize, Real, Space, Study

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
    from hola_opt import Minimize, Real, Space, Study

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
    from hola_opt import Minimize, Real, Space, Study

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
    from hola_opt import Minimize, Real, Space, Study

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


# ==========================================================================
# Bounded leaderboard
# ==========================================================================


def test_max_leaderboard_size_caps_retained_trials():
    """A Study built with max_leaderboard_size retains at most that many trials.

    Discriminates the opt-in cap from the default (unbounded) behavior: both
    studies run the same number of trials, but only the bounded one evicts. If
    max_leaderboard_size were not threaded into StudyConfig (i.e. left as None),
    the bounded study would also retain every trial and this test would fail.
    """
    from hola_opt import Minimize, Real, Space, Study

    cap = 5
    n = 20

    bounded = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
        seed=0,
        max_leaderboard_size=cap,
    )
    unbounded = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
        seed=0,
    )

    for study in (bounded, unbounded):
        for i in range(n):
            trial = study.ask()
            # Strictly improving losses so the just-told trial is never the
            # eviction victim and tell() always succeeds.
            study.tell(trial.trial_id, {"loss": float(n - i)})

    assert bounded.trial_count() == cap
    assert unbounded.trial_count() == n


def test_max_leaderboard_size_rejects_zero():
    """max_leaderboard_size must be >= 1; zero is rejected at construction."""
    from hola_opt import Minimize, Real, Space, Study

    with pytest.raises(ValueError, match="at least 1"):
        Study(
            space=Space(x=Real(0.0, 1.0)),
            objectives=[Minimize("loss")],
            max_leaderboard_size=0,
        )


# ==========================================================================
# 12. GIL release: concurrent ask/tell from multiple Python threads
# ==========================================================================


def test_concurrent_ask_tell_no_deadlock_correct_count():
    """Drive ask/tell from several Python threads against one local Study.

    This verifies NO-DEADLOCK and CORRECT ACCOUNTING under concurrent access:
    many threads must make progress without deadlocking, and every successful
    tell must be reflected in the trial count. It does not by itself prove GIL
    release (the engine's accounting is lock-protected in Rust, so the count
    would be correct even without py.detach); GIL release for the engine path
    is verified by inspection, and for the foreground server by
    test_foreground_serve_releases_gil. No wall-clock timing is asserted (that
    would be flaky); the test only checks for liveness and correct accounting.
    """
    import threading

    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    n_threads = 8
    per_thread = 25
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(per_thread):
                trial = study.ask()
                study.tell(trial.trial_id, {"loss": trial.params["x"] ** 2})
        except BaseException as exc:  # noqa: BLE001 - record for the main thread
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    # A generous join timeout turns a deadlock into a test failure rather
    # than a hung suite.
    for t in threads:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in threads), "threads did not finish (possible deadlock)"
    assert not errors, f"worker threads raised: {errors!r}"
    assert study.trial_count() == n_threads * per_thread

    # Other GIL-releasing readers still return coherent results afterwards.
    assert len(study.top_k(5)) == 5
    assert len(study.trials()) == n_threads * per_thread


def test_foreground_serve_releases_gil(free_port):
    """Foreground serve() must release the GIL for the server's lifetime.

    serve(background=False) blocks the calling thread on the running server.
    Without py.detach the GIL is held for the whole server lifetime, freezing
    every other Python thread. Here the foreground server runs on a background
    daemon thread while the MAIN thread does ordinary Python work (ask/tell on
    a SEPARATE local Study), which must run concurrently rather than block on
    the GIL held by serve().

    A C-level faulthandler watchdog ensures the test can never hang the suite:
    a GIL-holding regression would freeze every Python thread (so a Python-level
    timeout could never fire), but faulthandler runs in C and hard-exits the
    process, turning a regression into a loud failure rather than an infinite hang.
    """
    import contextlib
    import faulthandler
    import threading
    import time

    from hola_opt import Minimize, Real, Space, Study

    # Hard-exit the process if anything below wedges the interpreter for 45s.
    faulthandler.dump_traceback_later(45, exit=True)

    server_study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    def run_server():
        # Blocks until the process exits; the daemon thread is reaped at exit.
        with contextlib.suppress(Exception):
            server_study.serve(port=free_port, background=False)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give the server a moment to take the runtime; if the GIL were held this
    # sleep itself could be delayed, but the discriminating check is below.
    time.sleep(0.2)

    # The key property: the main thread can run Python while the foreground
    # server is up. Drive ask/tell on a separate local Study within a generous
    # bound; if serve() held the GIL this would block.
    local = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])

    done = threading.Event()

    def main_work():
        for _ in range(20):
            t = local.ask()
            local.tell(t.trial_id, {"loss": t.params["x"] ** 2})
        done.set()

    worker = threading.Thread(target=main_work)
    worker.start()
    worker.join(timeout=30)

    # join() returned, so the interpreter is no longer at risk of wedging;
    # disarm the watchdog before asserting (a plain assertion failure must not
    # leave it armed to hard-exit the suite later).
    faulthandler.cancel_dump_traceback_later()

    assert done.is_set(), "main-thread Python work did not progress (GIL held by serve())"
    assert local.trial_count() == 20
