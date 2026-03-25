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
        proc.wait(timeout=5)


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


# ==========================================================================
# Worker Tests
# ==========================================================================


def test_worker_completes_trials(cli_binary, tmp_path):
    port = _find_free_port()
    config_path = write_yaml_config(tmp_path)
    url = f"http://localhost:{port}"

    server = subprocess.Popen(
        [cli_binary, "serve", str(config_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert _wait_for_server(url), "Server did not start"

        worker = subprocess.Popen(
            [
                cli_binary,
                "worker",
                "--server",
                url,
                "--mode",
                "exec",
                "--exec",
                "echo '{\"loss\": 0.5}'",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            assert body["trial_count"] >= 3, f"Worker completed only {body['count']} trials"
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert _wait_for_server(url), "Server did not start"

        # Worker script reads HOLA_PARAMS and uses param value as loss
        exec_cmd = (
            'python3 -c "'
            "import os, json; "
            "p = json.loads(os.environ['HOLA_PARAMS']); "
            "print(json.dumps({'loss': p['x']}))"
            '"'
        )

        worker = subprocess.Popen(
            [cli_binary, "worker", "--server", url, "--mode", "exec", "--exec", exec_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
