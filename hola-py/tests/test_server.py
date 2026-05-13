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
REST API and Study.connect() integration tests.

Each test starts a real HOLA server process and interacts via HTTP or
Study.connect(). Tests cover: ask/tell/top_k flow, trials listing,
error handling (unknown trial, double tell), sequential IDs, extra metric
fields, checkpoint save, multi-param space/objectives endpoints, objective
updates, and Study.connect() ask/tell/top_k/connection-error.
"""

import json
import os
import socket
import subprocess

import pytest
from conftest import _wait_for_server, http_json

from hola_opt import Minimize, Real, Space, Study

# ==========================================================================
# REST API Endpoint Tests
# ==========================================================================


class TestRestEndpoints:
    """Tests using a simple 1D continuous space server."""

    @pytest.fixture(autouse=True)
    def _server(self, running_server):
        self.url = running_server

    def test_ask_returns_trial(self):
        status, body = http_json(f"{self.url}/api/ask", method="POST")
        assert status == 200
        assert "trial_id" in body
        assert "params" in body
        assert isinstance(body["trial_id"], int)
        assert "x" in body["params"]

    def test_ask_tell_best_flow(self):
        # Ask
        _, trial = http_json(f"{self.url}/api/ask", method="POST")
        trial_id = trial["trial_id"]

        # Tell
        status, resp = http_json(
            f"{self.url}/api/tell",
            method="POST",
            body={"trial_id": trial_id, "metrics": {"loss": 0.42}},
        )
        assert status == 200
        assert resp["status"] == "ok"
        assert resp["trial_count"] == 1

        # Top K (replacement for /api/best)
        status, best = http_json(f"{self.url}/api/top_k?k=1")
        assert status == 200
        assert isinstance(best, list)
        assert len(best) == 1
        assert best[0]["trial_id"] == trial_id
        assert isinstance(best[0]["score_vector"], dict)

    def test_trials_empty(self):
        status, body = http_json(f"{self.url}/api/trials?sorted_by=index&include_infeasible=true")
        assert status == 200
        assert isinstance(body, list)
        assert body == []

    def test_trials_after_trials(self):
        for _ in range(3):
            _, trial = http_json(f"{self.url}/api/ask", method="POST")
            http_json(
                f"{self.url}/api/tell",
                method="POST",
                body={"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}},
            )
        status, body = http_json(f"{self.url}/api/trials?sorted_by=index&include_infeasible=true")
        assert status == 200
        assert isinstance(body, list)
        assert len(body) == 3

    def test_tell_unknown_trial_400(self):
        status, body = http_json(
            f"{self.url}/api/tell",
            method="POST",
            body={"trial_id": 999, "metrics": {"loss": 0.5}},
        )
        assert status == 400
        assert "error" in body

    def test_double_tell_400(self):
        _, trial = http_json(f"{self.url}/api/ask", method="POST")
        tid = trial["trial_id"]
        http_json(
            f"{self.url}/api/tell",
            method="POST",
            body={"trial_id": tid, "metrics": {"loss": 0.5}},
        )
        status, body = http_json(
            f"{self.url}/api/tell",
            method="POST",
            body={"trial_id": tid, "metrics": {"loss": 0.3}},
        )
        assert status == 400

    def test_sequential_ask_ids(self):
        ids = []
        for _ in range(5):
            _, trial = http_json(f"{self.url}/api/ask", method="POST")
            ids.append(trial["trial_id"])
        assert ids == [0, 1, 2, 3, 4]

    def test_tell_extra_fields_ok(self):
        _, trial = http_json(f"{self.url}/api/ask", method="POST")
        status, _ = http_json(
            f"{self.url}/api/tell",
            method="POST",
            body={
                "trial_id": trial["trial_id"],
                "metrics": {"loss": 0.3, "accuracy": 0.9, "latency": 50.0},
            },
        )
        assert status == 200

    def test_checkpoint_save(self):
        # Complete a trial first
        _, trial = http_json(f"{self.url}/api/ask", method="POST")
        http_json(
            f"{self.url}/api/tell",
            method="POST",
            body={"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}},
        )

        status, body = http_json(
            f"{self.url}/api/checkpoint/save",
            method="POST",
            body={"path": "test_checkpoint.json"},
        )
        assert status == 200
        assert body["status"] == "ok"
        assert body["checkpoint_type"] == "full"
        assert os.path.exists(body["path"])
        restored = Study.load(body["path"])
        assert restored.trial_count() == 1

    def test_checkpoint_save_rejects_absolute_path(self):
        status, body = http_json(
            f"{self.url}/api/checkpoint/save",
            method="POST",
            body={"path": "/tmp/hola_escape.json"},
        )
        assert status == 400
        assert "relative" in body["error"]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _write_sobol_server_config(tmp_path, *, load_from=None):
    load_from_line = ""
    if load_from is not None:
        load_from_line = f"  load_from: {json.dumps(str(load_from))}\n"
    config_path = tmp_path / ("loaded.yaml" if load_from else "study.yaml")
    config_path.write_text(
        "space:\n"
        "  x:\n"
        "    type: real\n"
        "    min: 0.0\n"
        "    max: 1.0\n"
        "objectives:\n"
        "  - field: loss\n"
        "    type: minimize\n"
        "    priority: 1.0\n"
        "strategy:\n"
        "  type: sobol\n"
        "  seed: 123\n"
        "checkpoint:\n"
        f"  directory: {json.dumps(str(tmp_path))}\n"
        "  interval: 50\n"
        "  max_checkpoints: 5\n"
        f"{load_from_line}",
        encoding="utf-8",
    )
    return config_path


def _start_server(cli_binary, config_path, port):
    url = f"http://localhost:{port}"
    proc = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not _wait_for_server(url):
        proc.kill()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        pytest.fail(f"Server failed to start within timeout. stderr: {stderr}")
    return proc, url


def _stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_cli_load_from_rest_full_checkpoint_preserves_sobol_sequence(
    cli_binary, free_port, tmp_path
):
    baseline = Study(
        space=Space(x=Real(0.0, 1.0)),
        objectives=[Minimize("loss")],
        strategy="sobol",
        seed=123,
    )
    for _ in range(3):
        baseline.ask()
    expected = baseline.ask()

    config_path = _write_sobol_server_config(tmp_path)
    proc, url = _start_server(cli_binary, config_path, free_port)
    try:
        for _ in range(3):
            status, trial = http_json(f"{url}/api/ask", method="POST")
            assert status == 200
            status, _ = http_json(
                f"{url}/api/tell",
                method="POST",
                body={
                    "trial_id": trial["trial_id"],
                    "metrics": {"loss": trial["params"]["x"]},
                },
            )
            assert status == 200

        status, body = http_json(
            f"{url}/api/checkpoint/save",
            method="POST",
            body={"path": "server-full.json", "description": "server full"},
        )
        assert status == 200
        assert body["checkpoint_type"] == "full"
        checkpoint_path = body["path"]
    finally:
        _stop_server(proc)

    restored = Study.load(checkpoint_path)
    assert restored.ask().params == expected.params

    loaded_config_path = _write_sobol_server_config(tmp_path, load_from=checkpoint_path)
    proc, loaded_url = _start_server(cli_binary, loaded_config_path, _free_port())
    try:
        status, trial = http_json(f"{loaded_url}/api/ask", method="POST")
        assert status == 200
        assert trial["trial_id"] == 3
        assert trial["params"] == expected.params
    finally:
        _stop_server(proc)


class TestRestMultiParam:
    """Tests using a multi-param server (continuous, discrete, categorical)."""

    @pytest.fixture(autouse=True)
    def _server(self, running_server):
        self.url = running_server

    @pytest.mark.server_config(
        space={
            "lr": {"type": "real", "min": 0.0001, "max": 0.1, "scale": "log10"},
            "layers": {"type": "integer", "min": 1, "max": 10},
            "opt": {"type": "categorical", "choices": ["adam", "sgd", "rmsprop"]},
        },
        objectives=[{"field": "loss", "type": "minimize", "priority": 1.0}],
        strategy={"type": "sobol", "refit_interval": 20},
    )
    def test_space_endpoint(self, running_server):
        url = running_server
        status, body = http_json(f"{url}/api/space")
        assert status == 200
        param_names = {p["name"] for p in body["params"]}
        assert "lr" in param_names
        assert "layers" in param_names
        assert "opt" in param_names

    @pytest.mark.server_config(
        objectives=[{"field": "loss", "type": "minimize", "priority": 1.0}],
    )
    def test_objectives_endpoint(self, running_server):
        url = running_server
        status, body = http_json(f"{url}/api/objectives")
        assert status == 200
        assert "objectives" in body
        assert len(body["objectives"]) >= 1

    @pytest.mark.server_config(
        objectives=[{"field": "loss", "type": "minimize", "priority": 1.0}],
    )
    def test_update_objectives(self, running_server):
        url = running_server
        # Do a trial first
        _, trial = http_json(f"{url}/api/ask", method="POST")
        http_json(
            f"{url}/api/tell",
            method="POST",
            body={"trial_id": trial["trial_id"], "metrics": {"loss": 0.5, "accuracy": 0.9}},
        )

        # Update objectives
        status, body = http_json(
            f"{url}/api/objectives",
            method="PATCH",
            body={"objectives": [{"field": "accuracy", "type": "maximize", "priority": 1.0}]},
        )
        assert status == 200
        assert body["status"] == "ok"


# ==========================================================================
# Study.connect() Live Integration Tests
# ==========================================================================


class TestStudyConnect:
    @pytest.fixture(autouse=True)
    def _server(self, running_server):
        self.url = running_server

    def test_study_connect_ask_tell_best(self):
        remote = Study.connect(self.url)
        t = remote.ask()
        assert t.trial_id == 0
        assert "x" in t.params

        remote.tell(t.trial_id, {"loss": 0.42})

        top = remote.top_k(1)
        assert len(top) == 1
        best = top[0]
        assert best.trial_id == t.trial_id
        assert isinstance(best.score_vector, dict)

    def test_study_connect_multiple_trials(self):
        remote = Study.connect(self.url)
        for _ in range(10):
            t = remote.ask()
            remote.tell(t.trial_id, {"loss": t.params["x"] ** 2})

        top = remote.top_k(1)
        assert len(top) == 1
        assert isinstance(top[0].score_vector, dict)

    def test_study_connect_trial_count(self):
        remote = Study.connect(self.url)
        assert remote.trial_count() == 0

        for _ in range(3):
            t = remote.ask()
            remote.tell(t.trial_id, {"loss": t.params["x"] ** 2})

        assert remote.trial_count() == 3

    def test_study_connect_trials(self):
        remote = Study.connect(self.url)
        for _ in range(3):
            t = remote.ask()
            remote.tell(t.trial_id, {"loss": t.params["x"] ** 2})

        trials = remote.trials()
        assert len(trials) == 3
        for trial in trials:
            assert hasattr(trial, "trial_id")
            assert isinstance(trial.params, dict)
            assert isinstance(trial.score_vector, dict)

    def test_study_connect_run(self):
        remote = Study.connect(self.url)
        result = remote.run(lambda p: {"loss": p["x"] ** 2}, n_trials=5, n_workers=1)
        assert result is remote  # returns self
        assert remote.trial_count() == 5
        top = remote.top_k(1)
        assert len(top) == 1

    @pytest.mark.server_config(
        objectives=[
            {"field": "loss", "type": "minimize", "target": 0.0, "limit": 5.0, "priority": 1.0},
            {
                "field": "latency",
                "type": "minimize",
                "target": 0.0,
                "limit": 100.0,
                "priority": 2.0,
            },
        ],
    )
    def test_study_connect_pareto_front(self, running_server):
        remote = Study.connect(running_server)
        for _ in range(5):
            t = remote.ask()
            remote.tell(t.trial_id, {"loss": t.params["x"], "latency": 1.0 - t.params["x"]})

        front = remote.pareto_front()
        assert isinstance(front, list)
        assert len(front) > 0
        for trial in front:
            assert hasattr(trial, "trial_id")
            assert isinstance(trial.params, dict)
            assert isinstance(trial.scores, dict)

    def test_study_connect_pareto_front_scalar_returns_empty(self):
        remote = Study.connect(self.url)
        # Default server is single-objective, should return empty list
        t = remote.ask()
        remote.tell(t.trial_id, {"loss": 0.5})
        front = remote.pareto_front()
        assert front == []

    def test_study_connect_connection_error(self):
        remote = Study.connect("http://localhost:1")
        with pytest.raises(ValueError):
            remote.ask()
