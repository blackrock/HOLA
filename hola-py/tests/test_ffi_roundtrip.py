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

Covers non-finite float (inf/-inf/nan) and large-u64 round-tripping through
tell()/CompletedTrial.metrics, Study.connect() URL validation, and the "ln"
natural-log scale alias on Real.
"""

import math

import pytest

# ==========================================================================
# Non-finite float round-trip through tell() / CompletedTrial.metrics
# ==========================================================================


def _completed_trial(metrics):
    """Run a single ask/tell and return the CompletedTrial for `metrics`."""
    from hola_opt import Minimize, Real, Space, Study

    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    t = study.ask()
    return study.tell(t.trial_id, metrics)


def test_metrics_inf_roundtrip():
    ct = _completed_trial({"loss": 0.5, "extra": float("inf")})
    assert ct.metrics["extra"] == float("inf")


def test_metrics_neg_inf_roundtrip():
    ct = _completed_trial({"loss": 0.5, "extra": float("-inf")})
    assert ct.metrics["extra"] == float("-inf")


def test_metrics_nan_roundtrip():
    ct = _completed_trial({"loss": 0.5, "extra": float("nan")})
    assert math.isnan(ct.metrics["extra"])


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
