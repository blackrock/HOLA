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
Shared fixtures for the HOLA test suite.

Provides reusable spaces, server lifecycle management, and HTTP helpers.
All server/CLI fixtures use only stdlib (no requests/pyyaml dependencies).
"""

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from hola_opt import Categorical, Integer, Real, Space

# ==========================================================================
# Reusable spaces
# ==========================================================================


@pytest.fixture
def simple_space():
    return Space(x=Real(0.0, 1.0))


@pytest.fixture
def sphere_space():
    return Space(x=Real(-5.0, 5.0), y=Real(-5.0, 5.0))


@pytest.fixture
def multi_param_space():
    return Space(
        lr=Real(1e-4, 0.1, scale="log10"),
        layers=Integer(1, 10),
        opt=Categorical(["adam", "sgd", "rmsprop"]),
    )


# ==========================================================================
# Port and binary helpers
# ==========================================================================


@pytest.fixture
def free_port():
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def cli_binary():
    """Build the CLI binary once per session and return its path."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run(
        ["cargo", "build", "-p", "hola-cli"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to build hola-cli: {result.stderr}")

    binary = os.path.join(project_root, "target", "debug", "hola")
    if not os.path.exists(binary):
        pytest.skip(f"CLI binary not found at {binary}")
    return binary


# ==========================================================================
# YAML config helpers
# ==========================================================================


def write_yaml_config(
    tmp_path,
    *,
    space=None,
    objectives=None,
    strategy=None,
):
    """Write a YAML config file and return the path.

    Uses simple string formatting to avoid pyyaml dependency.
    """
    if space is None:
        space = {"x": {"type": "continuous", "min": 0.0, "max": 1.0}}

    lines = ["space:"]
    for name, cfg in space.items():
        lines.append(f"  {name}:")
        for k, v in cfg.items():
            if k == "choices":
                lines.append(f"    {k}:")
                assert isinstance(v, (list, tuple))
                for c in v:
                    lines.append(f"      - {c}")
            else:
                lines.append(f"    {k}: {v}")

    if objectives is None:
        objectives = [{"field": "loss", "type": "minimize", "priority": 1.0}]
    lines.append("objectives:")
    for obj in objectives:
        lines.append(f"  - field: {obj['field']}")
        lines.append(f"    type: {obj['type']}")
        if "target" in obj:
            lines.append(f"    target: {obj['target']}")
        if "limit" in obj:
            lines.append(f"    limit: {obj['limit']}")
        if "priority" in obj:
            lines.append(f"    priority: {obj['priority']}")

    if strategy:
        lines.append("strategy:")
        for k, v in strategy.items():
            lines.append(f"  {k}: {v}")

    config_path = os.path.join(str(tmp_path), "study.yaml")
    with open(config_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return config_path


@pytest.fixture
def yaml_config(tmp_path):
    """Factory fixture: call with kwargs to create a YAML config file."""

    def _factory(**kwargs):
        return write_yaml_config(tmp_path, **kwargs)

    return _factory


# ==========================================================================
# HTTP helpers
# ==========================================================================


def http_json(url, method="GET", body=None):
    """Make an HTTP request and return (status_code, json_body).

    Uses urllib only — no extra dependencies.
    """
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            body_json = json.loads(raw)
        except json.JSONDecodeError:
            body_json = {"raw": raw}
        return e.code, body_json


# ==========================================================================
# Server fixture
# ==========================================================================


def _wait_for_server(url, timeout=10):
    """Poll GET /api/space until the server is ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = http_json(f"{url}/api/space")
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


@pytest.fixture
def running_server(cli_binary, free_port, tmp_path, request):
    """Start a HOLA server and yield its base URL.

    By default uses a simple 1D continuous space. Pass server_config marker
    to customize.
    """
    marker = request.node.get_closest_marker("server_config")
    config_kwargs = marker.kwargs if marker else {}

    config_path = write_yaml_config(tmp_path, **config_kwargs)
    port = free_port
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

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ==========================================================================
# Markers
# ==========================================================================


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "server_config: pass kwargs to configure the test server")
