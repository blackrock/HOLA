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
FFI round-trip tests for the HOLA Python <-> JSON conversion layer.

Covers non-finite score decoding without retyping literal raw JSON strings,
large-u64 round-tripping through tell()/CompletedTrial.metrics, Study.connect()
URL validation, and the "ln" natural-log scale alias on Real.
"""

import math

import pytest

# ==========================================================================
# Raw JSON preservation and non-finite numeric score decoding
# ==========================================================================


def _completed_trial(metrics):
    """Run a single ask/tell and return the CompletedTrial for `metrics`."""
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t = study.ask()
    return study.tell(t.trial_id, metrics)


def test_raw_metrics_inf_uses_json_sentinel_without_retyping_literal_strings():
    ct = _completed_trial({"loss": 0.5, "extra": float("inf")})
    assert ct.metrics["extra"] == "inf"


def test_raw_metrics_neg_inf_uses_json_sentinel():
    ct = _completed_trial({"loss": 0.5, "extra": float("-inf")})
    assert ct.metrics["extra"] == "-inf"


def test_raw_metrics_nan_uses_json_sentinel():
    ct = _completed_trial({"loss": 0.5, "extra": float("nan")})
    assert ct.metrics["extra"] == "nan"


def test_literal_nonfinite_sentinel_strings_are_preserved_in_raw_metrics():
    ct = _completed_trial(
        {
            "loss": 0.5,
            "positive": "inf",
            "negative": "-inf",
            "not_a_number": "nan",
        }
    )
    assert ct.metrics["positive"] == "inf"
    assert ct.metrics["negative"] == "-inf"
    assert ct.metrics["not_a_number"] == "nan"


def test_literal_nonfinite_sentinel_categorical_params_remain_strings():
    from hola_opt import Categorical, Minimize, Space, Study

    for choice in ("inf", "-inf", "nan"):
        study = Study(
            space=Space(label=Categorical([choice])),
            objectives=[Minimize("loss")],
            strategy="random",
            seed=7,
        )
        trial = study.ask()
        assert trial.params["label"] == choice
        assert isinstance(trial.params["label"], str)
        completed = study.tell(trial.trial_id, {"loss": 0.5})
        assert completed.params["label"] == choice
        assert isinstance(completed.params["label"], str)


def test_nonfinite_score_sentinels_decode_only_in_numeric_score_fields():
    ct = _completed_trial({"loss": float("inf")})
    assert ct.metrics["loss"] == "inf"
    assert math.isinf(ct.scores["loss"])
    assert math.isinf(ct.score_vector["loss"])


def test_nonfinite_objective_metrics_keep_sign_and_direction_during_scoring():
    negative = _completed_trial({"loss": float("-inf")})
    assert negative.scores["loss"] == float("-inf")
    assert negative.score_vector["loss"] == float("-inf")

    nan = _completed_trial({"loss": float("nan")})
    assert math.isnan(nan.scores["loss"])
    assert math.isnan(nan.score_vector["loss"])

    from hola_opt import Maximize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Maximize("reward")])
    trial = study.ask()
    positive_reward = study.tell(trial.trial_id, {"reward": float("inf")})
    assert positive_reward.scores["reward"] == float("-inf")


def test_metrics_large_u64_roundtrip():
    """A u64 value larger than i64::MAX keeps integer precision."""
    big = 2**63 + 7  # > i64::MAX (2**63 - 1), still < 2**64
    ct = _completed_trial({"loss": 0.5, "huge": big})
    assert ct.metrics["huge"] == big
    assert isinstance(ct.metrics["huge"], int)


def test_metrics_finite_float_still_number():
    ct = _completed_trial({"loss": 0.5, "ok": 1.25})
    assert ct.metrics["ok"] == 1.25


# ==========================================================================
# Study.connect() URL validation (no network handshake)
# ==========================================================================


def test_connect_rejects_non_url():
    from hola_opt import Study

    with pytest.raises(ValueError):
        Study.connect("not-a-url")


def test_connect_rejects_non_http_scheme():
    from hola_opt import Study

    with pytest.raises(ValueError):
        Study.connect("ftp://example.com")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com?tenant=one",
        "https://example.com/#dashboard",
        "https://user:password@example.com",
    ],
)
def test_connect_rejects_ambiguous_base_url_components(url):
    from hola_opt import ConfigurationError, Study

    with pytest.raises(ConfigurationError):
        Study.connect(url)


def test_connect_valid_http_no_network():
    """A valid http URL succeeds with no network request made."""
    from hola_opt import Study

    # An unreachable host is fine: connect() must not contact it.
    remote = Study.connect("http://localhost:9999")
    assert remote is not None


def test_connect_valid_https_no_network():
    from hola_opt import Study

    remote = Study.connect("https://example.invalid:8443")
    assert remote is not None


# ==========================================================================
# Real "ln" natural-log scale alias
# ==========================================================================


def test_real_scale_ln_accepted():
    from hola_opt import Real

    r = Real(min=1e-4, max=0.1, scale="ln")
    assert r.scale == "ln"


def test_real_scale_ln_usable_in_study():
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(1e-4, 0.1, scale="ln")), objectives=[Minimize("loss")])
    t = study.ask()
    assert 1e-4 <= t.params["x"] <= 0.1
    study.tell(t.trial_id, {"loss": 0.5})
    assert study.trial_count() == 1


def test_real_scale_invalid_still_rejected():
    from hola_opt import Real

    with pytest.raises(ValueError):
        Real(min=0.0, max=1.0, scale="bogus")
