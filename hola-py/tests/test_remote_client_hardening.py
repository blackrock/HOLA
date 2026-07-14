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

"""Focused regressions for the hardened Study.connect HTTP client."""

import gc
import json
import math
import os
import threading
import time
import warnings
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pytest

from hola_opt import ConfigurationError, Minimize, Real, RemoteError, Space, Study


class _JsonHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep expected test-server failures out of pytest output."""

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # Expected when a request-timeout regression test closes early.
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)


@contextmanager
def _running_http_server(handler_type):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _SlowAskHandler(_JsonHandler):
    request_started = threading.Event()

    def do_POST(self):  # noqa: N802 - stdlib handler API
        type(self).request_started.set()
        time.sleep(0.4)
        self.send_json(200, {"trial_id": 1, "params": {"x": 0.5}})


def test_remote_request_timeout_is_configurable_and_bounded():
    _SlowAskHandler.request_started = threading.Event()
    with _running_http_server(_SlowAskHandler) as url:
        remote = Study.connect(url, request_timeout=0.1)
        started = time.monotonic()
        with pytest.raises(RemoteError, match="timed out"):
            remote.ask()
        assert time.monotonic() - started < 1.0
        assert _SlowAskHandler.request_started.is_set(), (
            "the Python server thread could not progress while the client waited"
        )


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("argument", ["connect_timeout", "request_timeout"])
def test_remote_timeouts_must_be_positive_and_finite(argument, value):
    with pytest.raises(ConfigurationError, match=argument):
        Study.connect("http://example.invalid", **{argument: value})


class _StatusErrorHandler(_JsonHandler):
    def do_GET(self):  # noqa: N802 - stdlib handler API
        self.send_json(503, {"error": "read path exploded"})


def test_remote_http_error_preserves_status_and_structured_detail():
    with _running_http_server(_StatusErrorHandler) as url:
        remote = Study.connect(url)
        with pytest.raises(RemoteError, match=r"HTTP 503 Service Unavailable: read path exploded"):
            remote.top_k(1)


class _MalformedAskHandler(_JsonHandler):
    payload = {"trial_id": 1}

    def do_POST(self):  # noqa: N802 - stdlib handler API
        self.send_json(200, type(self).payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"trial_id": 1},
        {"trial_id": 1, "params": None},
        {"trial_id": 1, "params": []},
        {"trial_id": 1, "params": "not-an-object"},
    ],
)
def test_remote_ask_rejects_missing_or_non_object_params(payload):
    _MalformedAskHandler.payload = payload
    with _running_http_server(_MalformedAskHandler) as url:
        remote = Study.connect(url)
        with pytest.raises(RemoteError, match="non-object 'params'"):
            remote.ask()


class _RetrySafeAskHandler(_JsonHandler):
    lock = threading.Lock()
    keys = []
    calls = 0

    def do_POST(self):  # noqa: N802 - stdlib handler API
        with type(self).lock:
            type(self).calls += 1
            call = type(self).calls
            type(self).keys.append(self.headers.get("Idempotency-Key"))
        if call == 1:
            time.sleep(0.2)
        self.send_json(200, {"trial_id": 41, "params": {"x": 0.25}})


def test_remote_ask_reuses_idempotency_key_after_uncertain_response():
    _RetrySafeAskHandler.keys = []
    _RetrySafeAskHandler.calls = 0
    with _running_http_server(_RetrySafeAskHandler) as url:
        remote = Study.connect(url, request_timeout=0.05)
        with pytest.raises(RemoteError, match="timed out"):
            remote.ask()
        trial = remote.ask()

    assert trial.trial_id == 41
    assert trial.params == {"x": 0.25}
    assert len(_RetrySafeAskHandler.keys) == 2
    assert _RetrySafeAskHandler.keys[0]
    assert _RetrySafeAskHandler.keys[0] == _RetrySafeAskHandler.keys[1]


class _SuccessfulAskKeysHandler(_JsonHandler):
    lock = threading.Lock()
    keys = []

    def do_POST(self):  # noqa: N802 - stdlib handler API
        with type(self).lock:
            trial_id = len(type(self).keys)
            type(self).keys.append(self.headers.get("Idempotency-Key"))
        self.send_json(200, {"trial_id": trial_id, "params": {"x": 0.5}})


def test_remote_successful_asks_rotate_idempotency_keys():
    _SuccessfulAskKeysHandler.keys = []
    with _running_http_server(_SuccessfulAskKeysHandler) as url:
        remote = Study.connect(url)
        assert remote.ask().trial_id == 0
        assert remote.ask().trial_id == 1

    assert len(_SuccessfulAskKeysHandler.keys) == 2
    assert all(_SuccessfulAskKeysHandler.keys)
    assert _SuccessfulAskKeysHandler.keys[0] != _SuccessfulAskKeysHandler.keys[1]


class _PathPrefixAskHandler(_JsonHandler):
    paths = []

    def do_POST(self):  # noqa: N802 - stdlib handler API
        type(self).paths.append(self.path)
        self.send_json(200, {"trial_id": 3, "params": {"x": 0.5}})


def test_remote_base_url_path_prefix_is_joined_canonically():
    _PathPrefixAskHandler.paths = []
    with _running_http_server(_PathPrefixAskHandler) as url:
        remote = Study.connect(f"{url}/tenant/hola///")
        assert remote.ask().trial_id == 3

    assert _PathPrefixAskHandler.paths == ["/tenant/hola/api/ask"]


_COMPLETED_TRIAL = {
    "trial_id": 7,
    "params": {"x": 0.5},
    "metrics": {"loss": 0.25},
    "scores": {"loss": 0.25},
    "score_vector": {"loss": 0.25},
    "rank": 0,
    "pareto_front": 0,
    "completed_at": 1,
}


class _AuthenticatedApiHandler(_JsonHandler):
    seen = []

    def authorized(self):
        authorization = self.headers.get("Authorization")
        type(self).seen.append((self.command, self.path, authorization))
        if authorization == "Bearer test-secret":
            return True
        self.send_json(401, {"error": "missing bearer token"})
        return False

    def do_POST(self):  # noqa: N802 - stdlib handler API
        if not self.authorized():
            return
        path = urlsplit(self.path).path
        if path == "/api/ask":
            self.send_json(200, {"trial_id": 7, "params": {"x": 0.5}})
        elif path == "/api/tell":
            # Omit `trial` to exercise both authenticated compatibility reads.
            self.send_json(200, {"status": "ok"})
        elif path == "/api/cancel":
            self.send_json(200, {"status": "ok"})
        elif path == "/api/heartbeat":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "trial_id": 7,
                    "lease_expires_at_ms": 1_900_000_000_000,
                },
            )
        else:
            self.send_json(404, {"error": "unknown endpoint"})

    def do_PATCH(self):  # noqa: N802 - stdlib handler API
        if not self.authorized():
            return
        self.send_json(200, {"status": "ok", "rescalarized_trials": 1})

    def do_GET(self):  # noqa: N802 - stdlib handler API
        if not self.authorized():
            return
        path = urlsplit(self.path).path
        if path == "/api/trial/7":
            # Older servers do not provide this endpoint; the client then reads
            # the authenticated trials collection as its final fallback.
            self.send_json(404, {"error": "single-trial endpoint unavailable"})
        elif path == "/api/trial_count":
            self.send_json(200, {"trial_count": 1})
        elif path in {"/api/top_k", "/api/pareto_front", "/api/trials"}:
            self.send_json(200, [_COMPLETED_TRIAL])
        else:
            self.send_json(404, {"error": "unknown endpoint"})


def test_bearer_token_is_applied_to_every_remote_endpoint_and_fallback():
    _AuthenticatedApiHandler.seen = []
    with _running_http_server(_AuthenticatedApiHandler) as url:
        remote = Study.connect(url, token="test-secret")
        trial = remote.ask()
        assert remote.tell(trial.trial_id, {"loss": 0.25}).trial_id == trial.trial_id
        assert len(remote.top_k(1)) == 1
        assert len(remote.pareto_front()) == 1
        assert len(remote.trials()) == 1
        assert remote.trial_count() == 1
        remote.update_objectives([Minimize("loss")])
        assert remote.heartbeat(7) == 1_900_000_000_000
        remote.cancel(99)

    assert _AuthenticatedApiHandler.seen
    assert all(auth == "Bearer test-secret" for _, _, auth in _AuthenticatedApiHandler.seen)
    paths = [urlsplit(path).path for _, path, _ in _AuthenticatedApiHandler.seen]
    assert paths.count("/api/trials") >= 2  # tell fallback plus explicit trials()
    assert "/api/trial/7" in paths
    assert "/api/heartbeat" in paths


class _WrongTellIdentityHandler(_JsonHandler):
    embedded = True

    def do_POST(self):  # noqa: N802 - stdlib handler API
        wrong_trial = {**_COMPLETED_TRIAL, "trial_id": 99}
        if type(self).embedded:
            self.send_json(200, {"status": "ok", "trial": wrong_trial})
        else:
            self.send_json(200, {"status": "ok"})

    def do_GET(self):  # noqa: N802 - stdlib handler API
        self.send_json(200, {**_COMPLETED_TRIAL, "trial_id": 99})


@pytest.mark.parametrize(
    ("embedded", "source"),
    [(True, "Tell acknowledgement"), (False, "Single-trial response")],
)
def test_remote_tell_rejects_wrong_trial_identity(embedded, source):
    _WrongTellIdentityHandler.embedded = embedded
    with _running_http_server(_WrongTellIdentityHandler) as url:
        remote = Study.connect(url)
        with pytest.raises(RemoteError, match=rf"{source} returned trial_id 99 instead of 7"):
            remote.tell(7, {"loss": 0.25})


class _TellWarningsHandler(_JsonHandler):
    payload = {
        "status": "ok",
        "trial": _COMPLETED_TRIAL,
        "post_commit_warnings": [],
    }

    def do_POST(self):  # noqa: N802 - stdlib handler API
        self.send_json(200, type(self).payload)


def test_remote_tell_surfaces_post_commit_warnings_without_changing_result():
    _TellWarningsHandler.payload = {
        "status": "ok",
        "trial": _COMPLETED_TRIAL,
        "post_commit_warnings": [
            "auto-checkpoint failed",
            "strategy refit failed",
        ],
    }
    with _running_http_server(_TellWarningsHandler) as url:
        remote = Study.connect(url)
        with pytest.warns(RuntimeWarning) as emitted:
            completed = remote.tell(7, {"loss": 0.25})

    assert completed.trial_id == 7
    assert [str(item.message) for item in emitted] == [
        "auto-checkpoint failed",
        "strategy refit failed",
    ]


def test_remote_tell_warning_as_error_policy_cannot_mask_committed_result(capfd):
    _TellWarningsHandler.payload = {
        "status": "ok",
        "trial": _COMPLETED_TRIAL,
        "post_commit_warnings": ["strategy refit failed"],
    }
    with _running_http_server(_TellWarningsHandler) as url:
        remote = Study.connect(url)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            completed = remote.tell(7, {"loss": 0.25})

    assert completed.trial_id == 7
    assert "post-commit warning: strategy refit failed" in capfd.readouterr().err


@pytest.mark.parametrize(
    ("post_commit_warnings", "message"),
    [
        ("failed", "non-array post_commit_warnings"),
        ([42], "non-string post_commit_warnings entry"),
        (["valid prefix", None], "non-string post_commit_warnings entry"),
    ],
)
def test_remote_tell_rejects_malformed_post_commit_warnings_without_emitting(
    post_commit_warnings, message
):
    _TellWarningsHandler.payload = {
        "status": "ok",
        "trial": _COMPLETED_TRIAL,
        "post_commit_warnings": post_commit_warnings,
    }
    with _running_http_server(_TellWarningsHandler) as url:
        remote = Study.connect(url)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            with pytest.raises(RemoteError, match=message):
                remote.tell(7, {"loss": 0.25})

    assert emitted == []


class _MalformedHeartbeatHandler(_JsonHandler):
    payload = {"status": "ok", "trial_id": 7}

    def do_POST(self):  # noqa: N802 - stdlib handler API
        self.send_json(200, type(self).payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"trial_id": 7, "lease_expires_at_ms": 99}, "status 'ok'"),
        ({"status": "ok", "lease_expires_at_ms": 99}, "trial_id"),
        (
            {"status": "ok", "trial_id": 8, "lease_expires_at_ms": 99},
            "instead of 7",
        ),
        ({"status": "ok", "trial_id": 7}, "lease_expires_at_ms"),
    ],
)
def test_remote_heartbeat_validates_response_schema(payload, message):
    _MalformedHeartbeatHandler.payload = payload
    with _running_http_server(_MalformedHeartbeatHandler) as url:
        remote = Study.connect(url)
        with pytest.raises(RemoteError, match=message):
            remote.heartbeat(7)


def test_local_study_rejects_distributed_heartbeat():
    study = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    trial = study.ask()
    with pytest.raises(ConfigurationError, match="only available for remote"):
        study.heartbeat(trial.trial_id)


class _MutationAcknowledgementHandler(_JsonHandler):
    cancel_payload = {"status": "ok"}
    update_payload = {"status": "ok", "rescalarized_trials": 0}

    def do_POST(self):  # noqa: N802 - stdlib handler API
        self.send_json(200, type(self).cancel_payload)

    def do_PATCH(self):  # noqa: N802 - stdlib handler API
        self.send_json(200, type(self).update_payload)


@pytest.mark.parametrize("payload", [{}, {"status": "error"}])
def test_remote_cancel_rejects_noncanonical_2xx_acknowledgement(payload):
    _MutationAcknowledgementHandler.cancel_payload = payload
    with _running_http_server(_MutationAcknowledgementHandler) as url:
        remote = Study.connect(url)
        with pytest.raises(RemoteError, match="canonical status 'ok'"):
            remote.cancel(7)


def test_remote_cancel_rejects_mismatched_optional_trial_identity():
    _MutationAcknowledgementHandler.cancel_payload = {"status": "ok", "trial_id": 8}
    with _running_http_server(_MutationAcknowledgementHandler) as url:
        remote = Study.connect(url)
        with pytest.raises(RemoteError, match="trial_id 8 instead of 7"):
            remote.cancel(7)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "canonical status 'ok'"),
        ({"status": "error", "rescalarized_trials": 0}, "canonical status 'ok'"),
        ({"status": "ok"}, "rescalarized_trials"),
    ],
)
def test_remote_objective_update_rejects_malformed_2xx_acknowledgement(payload, message):
    _MutationAcknowledgementHandler.update_payload = payload
    with _running_http_server(_MutationAcknowledgementHandler) as url:
        remote = Study.connect(url)
        with pytest.raises(RemoteError, match=message):
            remote.update_objectives([Minimize("loss")])


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="requires Linux /proc")
def test_many_studies_share_one_bounded_runtime_worker_pool():
    def runtime_worker_count():
        count = 0
        for task in os.scandir("/proc/self/task"):
            try:
                with open(f"{task.path}/comm", encoding="utf-8") as comm:
                    count += comm.read().strip() == "hola-py-worker"
            except FileNotFoundError:
                pass
        return count

    first = Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])
    time.sleep(0.05)
    before = runtime_worker_count()
    assert before > 0

    studies = [
        Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")]) for _ in range(24)
    ]
    studies.extend(Study.connect("http://example.invalid") for _ in range(24))
    time.sleep(0.05)

    assert runtime_worker_count() == before

    del studies, first
    gc.collect()
    assert runtime_worker_count() == before  # process-owned pool, no per-Study leak
