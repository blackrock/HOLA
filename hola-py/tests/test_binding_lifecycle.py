# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Regression tests for native conversion and Python-hosted worker lifecycle."""

import json
import math
import os
import subprocess
import sys
import textwrap
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Literal

import pytest

import hola_opt
from hola_opt import ConfigurationError, HolaError, Minimize, Real, RemoteError, Space, Study


def _study(**kwargs):
    return Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
        strategy="sobol",
        seed=1,
        **kwargs,
    )


def _connect_with_timeout(
    name: Literal["connect_timeout", "request_timeout"], value: float
) -> Study:
    if name == "connect_timeout":
        return Study.connect("http://127.0.0.1:8000", connect_timeout=value)
    return Study.connect("http://127.0.0.1:8000", request_timeout=value)


@pytest.mark.parametrize("name", ["connect_timeout", "request_timeout"])
@pytest.mark.parametrize("value", [1e100, 1e19, 1e-100])
def test_timeout_conversion_rejects_unrepresentable_values(name, value):
    with pytest.raises(ConfigurationError):
        _connect_with_timeout(name, value)


@pytest.mark.parametrize("name", ["connect_timeout", "request_timeout"])
def test_timeout_conversion_accepts_portable_upper_bound(name):
    # Construct the native reqwest client and runtime without a network request.
    remote = _connect_with_timeout(name, 3_153_600_000.0)
    assert isinstance(remote, Study)


@pytest.mark.parametrize("name", ["connect_timeout", "request_timeout"])
def test_timeout_conversion_rejects_one_float_step_above_portable_upper_bound(name):
    value = math.nextafter(3_153_600_000.0, math.inf)
    with pytest.raises(ConfigurationError, match=f"{name} must not exceed 3153600000 seconds"):
        _connect_with_timeout(name, value)


def test_cyclic_and_deep_metrics_are_recoverable_in_a_subprocess():
    # A native stack overflow kills the interpreter, so keep this regression
    # isolated and disable core files in the child before touching the binding.
    code = textwrap.dedent(
        """
        import os
        if os.name == 'posix':
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        from hola_opt import Study, Space, Real, Minimize, ObjectiveError

        cyclic_dict = {'loss': 0.5}
        cyclic_dict['metadata'] = cyclic_dict
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        deeply_nested = {}
        for _ in range(100):
            deeply_nested = {'child': deeply_nested}
        invalid = [cyclic_dict, {'loss': 0.5, 'metadata': cyclic_list},
                   {'loss': 0.5, 'metadata': deeply_nested}]
        for metrics in invalid:
            study = Study(space=Space(x=Real(0, 1)), objectives=[Minimize('loss')],
                          strategy='sobol', seed=1)
            trial = study.ask()
            try:
                study.tell(trial.trial_id, metrics)
            except ObjectiveError:
                pass
            else:
                raise AssertionError('invalid metrics accepted')
            assert study.trial_count() == 0
            study.tell(trial.trial_id, {'loss': 0.5})
            assert study.trial_count() == 1
        print('all conversion failures recovered without committing')
        """
    )
    env = dict(os.environ)
    package_parent = str(Path(hola_opt.__file__).parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_parent, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20, env=env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all conversion failures recovered" in result.stdout


def test_shared_acyclic_metadata_is_accepted():
    study = _study()
    trial = study.ask()
    shared = {"values": [1, 2, 3]}
    completed = study.tell(trial.trial_id, {"loss": 0.5, "left": shared, "right": shared})
    assert completed.metrics["left"] == completed.metrics["right"] == shared


def test_background_startup_conflict_stop_and_port_reuse(free_port):
    first, second = _study(), _study()
    try:
        first.serve(port=free_port, background=True)
        # No startup sleep: success means the bound listener is ready.
        remote = Study.connect(f"http://127.0.0.1:{free_port}")
        assert remote.trial_count() == 0
        with pytest.raises(HolaError, match="Server error"):
            second.serve(port=free_port, background=True)
        with pytest.raises(ConfigurationError, match="already has a background server"):
            first.serve(port=free_port, background=True)
        first.stop()
        first.stop()  # Idempotent when no server is running.
        second.serve(port=free_port, background=True)
        assert Study.connect(f"http://127.0.0.1:{free_port}").trial_count() == 0
    finally:
        first.stop()
        second.stop()


def test_network_host_requires_authentication(free_port):
    study = _study()
    with pytest.raises(HolaError, match="auth token is required"):
        study.serve(port=free_port, background=True, host="0.0.0.0")
    try:
        study.serve(port=free_port, background=True, host="0.0.0.0", auth_token="test-only-token")
        remote = Study.connect(f"http://127.0.0.1:{free_port}", token="test-only-token")
        trial = remote.ask()
        remote.tell(trial.trial_id, {"loss": 0.5})
        assert remote.trial_count() == 1
    finally:
        study.stop()


@pytest.mark.parametrize("workers", [1, 2])
def test_remote_run_renews_short_leases(free_port, workers):
    study = _study()
    try:
        study.serve(port=free_port, background=True, lease_seconds=0.3)
        remote = Study.connect(f"http://127.0.0.1:{free_port}")

        def objective(params):
            time.sleep(0.8)
            return {"loss": params["x"]}

        assert remote.run(objective, n_trials=2, n_workers=workers) is remote
        assert remote.trial_count() == study.trial_count() == 2
    finally:
        study.stop()


@pytest.mark.parametrize("workers", [1, 2])
def test_remote_run_cancels_after_objective_failure(free_port, workers):
    study = _study(max_trials=2)
    try:
        study.serve(port=free_port, background=True, lease_seconds=0.3)
        remote = Study.connect(f"http://127.0.0.1:{free_port}")

        def objective(_params):
            time.sleep(0.5)
            raise RuntimeError("objective failed after lease renewal")

        with pytest.raises(RuntimeError, match="objective failed"):
            remote.run(objective, n_trials=2, n_workers=workers)
        assert remote.trial_count() == 0
        # Cancellation freed every allocation; no abandoned renewal task keeps
        # a hidden pending trial consuming this bounded study's capacity.
        next_trials = [remote.ask(), remote.ask()]
        for trial in next_trials:
            remote.cancel(trial.trial_id)
    finally:
        study.stop()


def test_remote_stop_is_rejected():
    with pytest.raises(ConfigurationError, match="local studies"):
        Study.connect("http://127.0.0.1:8000").stop()


def test_remote_run_retries_renewal_and_uses_server_duration():
    class LeaseHandler(BaseHTTPRequestHandler):
        heartbeats = 0
        committed = False

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def reply(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(encoded)

        def do_POST(self):
            if self.path == "/api/ask":
                self.reply(200, {"trial_id": 0, "params": {"x": 0.5}})
            elif self.path == "/api/heartbeat":
                type(self).heartbeats += 1
                if type(self).heartbeats == 2:
                    self.reply(503, {"error": "transient renewal failure"})
                    return
                # A deliberately skewed absolute deadline must not override
                # the server-supplied lease duration.
                self.reply(
                    200,
                    {
                        "status": "ok",
                        "trial_id": 0,
                        "lease_expires_at_ms": 1,
                        "lease_duration_ms": 300,
                    },
                )
            elif self.path == "/api/tell":
                type(self).committed = True
                self.reply(
                    200,
                    {
                        "status": "ok",
                        "trial": {
                            "trial_id": 0,
                            "params": {"x": 0.5},
                            "metrics": {"loss": 0.5},
                            "scores": {"loss": 0.5},
                            "score_vector": {"loss": 0.5},
                            "rank": 0,
                            "pareto_front": 0,
                            "completed_at": 1,
                        },
                    },
                )
            else:
                self.reply(404, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), LeaseHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = Study.connect(f"http://127.0.0.1:{server.server_port}")
        remote.run(lambda _params: (time.sleep(0.8), {"loss": 0.5})[1], n_trials=1)
        assert LeaseHandler.committed
        assert LeaseHandler.heartbeats >= 4
        after_completion = LeaseHandler.heartbeats
        time.sleep(0.3)
        assert LeaseHandler.heartbeats == after_completion
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("heartbeat_status", [404, 405, 200])
def test_remote_run_legacy_heartbeat_and_initial_timeout_cleanup(heartbeat_status):
    class LegacyHandler(BaseHTTPRequestHandler):
        heartbeats = 0
        cancelled = []

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def reply(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(encoded)

        def do_POST(self):
            if self.path == "/api/ask":
                self.reply(200, {"trial_id": 0, "params": {"x": 0.5}})
            elif self.path == "/api/heartbeat":
                type(self).heartbeats += 1
                if heartbeat_status == 200:
                    time.sleep(0.2)  # Exceeds this client's initial request timeout.
                    self.reply(
                        200,
                        {
                            "status": "ok",
                            "trial_id": 0,
                            "lease_expires_at_ms": 1,
                            "lease_duration_ms": 300,
                        },
                    )
                else:
                    self.reply(heartbeat_status, {})
            elif self.path == "/api/cancel":
                type(self).cancelled.append(0)
                self.reply(200, {"status": "ok", "trial_id": 0})
            elif self.path == "/api/tell":
                self.reply(
                    200,
                    {
                        "status": "ok",
                        "trial": {
                            "trial_id": 0,
                            "params": {"x": 0.5},
                            "metrics": {"loss": 0.5},
                            "scores": {"loss": 0.5},
                            "score_vector": {"loss": 0.5},
                            "rank": 0,
                            "pareto_front": 0,
                            "completed_at": 1,
                        },
                    },
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), LegacyHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evaluated = []
    try:
        remote = Study.connect(f"http://127.0.0.1:{server.server_port}", request_timeout=0.05)

        def objective(_params):
            evaluated.append(True)
            return {"loss": 0.5}

        if heartbeat_status == 200:
            with pytest.raises(RemoteError, match="timed out"):
                remote.run(objective, n_trials=1)
            assert not evaluated
            assert LegacyHandler.cancelled == [0]
        else:
            remote.run(objective, n_trials=1)
            assert evaluated == [True]
            assert not LegacyHandler.cancelled
        assert LegacyHandler.heartbeats == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
