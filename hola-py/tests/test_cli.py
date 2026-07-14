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
CLI binary tests.

Tests the hola-cli binary: --help output, serve startup and HTTP
response, bad/invalid config error exits, custom port binding, worker
trial completion, and HOLA_PARAMS environment variable passing.
"""

import os
import subprocess
import sys
import time

from conftest import _wait_for_server, http_json, write_yaml_config

# ==========================================================================
# Helpers
# ==========================================================================


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _python_command(tmp_path, name, source):
    """Write a worker program and return one shell command for every OS."""
    script = tmp_path / name
    script.write_text(source)
    return f'"{sys.executable}" "{script}"'


# ==========================================================================
# Basic CLI Tests
# ==========================================================================


def test_help_flag(cli_binary):
    result = subprocess.run(
        [cli_binary, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout.lower() or "serve" in result.stderr.lower()
    assert "worker" in result.stdout.lower() or "worker" in result.stderr.lower()


def test_serve_starts_responds(cli_binary, tmp_path):
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"
    logs = ""

    proc = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert _wait_for_server(url), "Server did not start"
        status, body = http_json(f"{url}/api/space")
        assert status == 200
    finally:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        logs = (stdout + stderr).decode("utf-8", errors="replace")
    assert "127.0.0.1" in logs
    assert str(port) in logs


def test_serve_bad_config_exits(cli_binary):
    result = subprocess.run(
        [cli_binary, "serve", "/nonexistent/path/config.yaml"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0


def test_serve_invalid_yaml_exits(cli_binary, tmp_path):
    bad_yaml = os.path.join(str(tmp_path), "bad.yaml")
    with open(bad_yaml, "w") as f:
        f.write("this: is: not: [valid yaml for hola")
    result = subprocess.run(
        [cli_binary, "serve", bad_yaml],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0


def test_serve_invalid_strategy_config_exits_with_helpful_error(cli_binary, tmp_path):
    config_path = write_yaml_config(tmp_path, strategy={"type": "soboll"})

    result = subprocess.run(
        [cli_binary, "serve", str(config_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Unknown strategy type 'soboll'" in result.stderr


def test_serve_invalid_scale_config_exits_with_parameter_name(cli_binary, tmp_path):
    config_path = write_yaml_config(
        tmp_path,
        space={"lr": {"type": "real", "min": 0.001, "max": 0.1, "scale": "log2"}},
    )

    result = subprocess.run(
        [cli_binary, "serve", str(config_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Parameter 'lr'" in result.stderr
    assert "unknown real scale" in result.stderr


def test_serve_custom_port(cli_binary, tmp_path):
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"

    proc = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert _wait_for_server(url), f"Server did not start on port {port}"
        status, _ = http_json(f"{url}/api/space")
        assert status == 200
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_serve_nonlocal_host_requires_token(cli_binary, tmp_path):
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)

    result = subprocess.run(
        [cli_binary, "serve", str(config_path), "--host", "0.0.0.0", "--port", str(port)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "--auth-token" in result.stderr


# ==========================================================================
# Worker Tests
# ==========================================================================


def test_worker_completes_trials(cli_binary, tmp_path):
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"

    server = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(url), "Server did not start"
        exec_cmd = _python_command(
            tmp_path,
            "constant_metrics.py",
            "import json\nprint(json.dumps({'loss': 0.5}))\n",
        )

        worker = subprocess.Popen(
            [
                cli_binary,
                "worker",
                "--server",
                url,
                "--mode",
                "exec",
                "--exec",
                exec_cmd,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            # Wait for the worker to complete some trials
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                _, body = http_json(f"{url}/api/trial_count")
                if body["trial_count"] >= 3:
                    break
                time.sleep(0.5)

            _, body = http_json(f"{url}/api/trial_count")
            assert body["trial_count"] >= 3, f"Worker completed only {body['trial_count']} trials"
        finally:
            worker.kill()
            worker.wait(timeout=5)
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_worker_receives_params_env(cli_binary, tmp_path):
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"

    server = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(url), "Server did not start"

        # Worker script reads HOLA_PARAMS and uses param value as loss.
        exec_cmd = _python_command(
            tmp_path,
            "params_metrics.py",
            "import json, os\n"
            "params = json.loads(os.environ['HOLA_PARAMS'])\n"
            "print(json.dumps({'loss': params['x']}))\n",
        )

        worker = subprocess.Popen(
            [cli_binary, "worker", "--server", url, "--mode", "exec", "--exec", exec_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                _, body = http_json(f"{url}/api/trials?sorted_by=index&include_infeasible=true")
                if len(body) >= 3:
                    break
                time.sleep(0.5)

            _, body = http_json(f"{url}/api/trials?sorted_by=index&include_infeasible=true")
            assert len(body) >= 3, f"Worker completed only {len(body)} trials"

            # Verify that trials have real score_vector values (not parse errors)
            for trial in body:
                assert isinstance(trial["score_vector"], dict)
                for v in trial["score_vector"].values():
                    assert isinstance(v, (int, float))
                    assert v < float("inf")
        finally:
            worker.kill()
            worker.wait(timeout=5)
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_worker_heartbeats_through_command_longer_than_server_lease(cli_binary, tmp_path):
    """A built-in worker must renew even the shortest supported server lease."""
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"

    server = subprocess.Popen(
        [
            cli_binary,
            "serve",
            str(config_path),
            "--port",
            str(port),
            "--lease-seconds",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(url), "Server did not start"
        exec_cmd = _python_command(
            tmp_path,
            "slow_metrics.py",
            "import json, time\ntime.sleep(2.2)\nprint(json.dumps({'loss': 0.25}))\n",
        )
        worker = subprocess.Popen(
            [cli_binary, "worker", "--server", url, "--mode", "exec", "--exec", exec_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                _, body = http_json(f"{url}/api/trial_count")
                if body["trial_count"] >= 1:
                    break
                time.sleep(0.2)

            _, body = http_json(f"{url}/api/trial_count")
            assert body["trial_count"] >= 1, (
                "Worker result was rejected after the one-second lease; "
                "periodic heartbeats did not keep it alive"
            )
        finally:
            worker.kill()
            stdout, stderr = worker.communicate(timeout=5)
            logs = (stdout + stderr).decode("utf-8", errors="replace")
            assert "permanent rejection" not in logs.lower()
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_callback_worker_heartbeats_through_command_longer_than_lease(cli_binary, tmp_path):
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"
    server = subprocess.Popen(
        [
            cli_binary,
            "serve",
            str(config_path),
            "--port",
            str(port),
            "--lease-seconds",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(url), "Server did not start"
        exec_cmd = _python_command(
            tmp_path,
            "slow_callback.py",
            "import json, os, time, urllib.request\n"
            "time.sleep(2.2)\n"
            "payload = json.dumps({'trial_id': int(os.environ['HOLA_TRIAL_ID']), "
            "'metrics': {'loss': 0.25}}).encode()\n"
            "request = urllib.request.Request(os.environ['HOLA_SERVER'] + '/api/tell', "
            "data=payload, headers={'Content-Type': 'application/json'})\n"
            "urllib.request.urlopen(request, timeout=5).read()\n",
        )
        worker = subprocess.Popen(
            [cli_binary, "worker", "--server", url, "--exec", exec_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                _, body = http_json(f"{url}/api/trial_count")
                if body["trial_count"] >= 1:
                    break
                time.sleep(0.2)
            _, body = http_json(f"{url}/api/trial_count")
            assert body["trial_count"] >= 1
        finally:
            worker.kill()
            stdout, stderr = worker.communicate(timeout=5)
            logs = (stdout + stderr).decode("utf-8", errors="replace")
            assert "lease heartbeat failed" not in logs.lower()
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_worker_exec_failure_cancels_not_reports(cli_binary, tmp_path):
    """A non-zero exit must cancel the trial, not POST a fake result.

    A failing command cancels the trial rather than reporting a synthetic
    metrics object that would scalarize to an infeasible (but completed)
    trial, so no trials are ever marked completed.
    """
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"

    server = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(url), "Server did not start"
        exec_cmd = _python_command(
            tmp_path,
            "failing_worker.py",
            "import sys\nsys.stderr.write('boom\\n')\nsys.exit(1)\n",
        )

        worker = subprocess.Popen(
            [
                cli_binary,
                "worker",
                "--server",
                url,
                "--mode",
                "exec",
                "--exec",
                exec_cmd,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            # Give the worker time to pull and fail several trials.
            time.sleep(5)
            _, body = http_json(f"{url}/api/trial_count")
            assert body["trial_count"] == 0, (
                "Failing exec command should cancel trials, "
                f"but {body['trial_count']} were completed"
            )
            _, trials = http_json(f"{url}/api/trials?sorted_by=index&include_infeasible=true")
            assert trials == [], (
                "Failing exec command must not report any (even infeasible) "
                f"trial results, got {trials}"
            )
        finally:
            worker.kill()
            worker.wait(timeout=5)
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_worker_exec_invalid_json_cancels(cli_binary, tmp_path):
    """A zero-exit command with non-JSON stdout must cancel, not report."""
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"

    server = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(url), "Server did not start"
        exec_cmd = _python_command(
            tmp_path,
            "invalid_metrics.py",
            "print('not-json-at-all')\n",
        )

        worker = subprocess.Popen(
            [
                cli_binary,
                "worker",
                "--server",
                url,
                "--mode",
                "exec",
                "--exec",
                exec_cmd,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(5)
            _, body = http_json(f"{url}/api/trial_count")
            assert body["trial_count"] == 0, (
                "Invalid-JSON exec output should cancel trials, "
                f"but {body['trial_count']} were completed"
            )
        finally:
            worker.kill()
            worker.wait(timeout=5)
    finally:
        server.terminate()
        server.wait(timeout=5)
