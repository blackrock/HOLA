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

"""Required server coverage must use the intended CLI and fail closed."""

import importlib.util
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load the sibling fixture explicitly so relocated wheel tests and type checking
# cannot resolve a different test suite's module named "conftest".
_fixture_spec = importlib.util.spec_from_file_location(
    "hola_cli_fixture_under_test", Path(__file__).with_name("conftest.py")
)
assert _fixture_spec is not None and _fixture_spec.loader is not None
conftest = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(conftest)


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture executable")
    path.chmod(0o755)
    return path


def test_supplied_cli_works_after_fixture_relocation(tmp_path, monkeypatch):
    binary = _executable(tmp_path / "packaged" / "hola")
    monkeypatch.setenv("HOLA_CLI_BINARY", str(binary))
    monkeypatch.setattr(conftest, "__file__", str(tmp_path / "isolated-tests" / "conftest.py"))
    monkeypatch.setattr(
        conftest.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Supplied CLI must not trigger a build"),
    )
    assert conftest.cli_binary.__wrapped__() == str(binary.resolve())


def test_missing_supplied_cli_fails_instead_of_skipping(tmp_path, monkeypatch):
    monkeypatch.setenv("HOLA_CLI_BINARY", str(tmp_path / "missing"))
    with pytest.raises(pytest.fail.Exception, match="Required CLI binary not found"):
        conftest.cli_binary.__wrapped__()


@pytest.mark.parametrize("absolute", [False, True])
def test_build_uses_locked_workspace_and_cargo_target_dir(tmp_path, monkeypatch, absolute):
    workspace = tmp_path / "checkout"
    fixture_path = workspace / "hola-py" / "tests" / "conftest.py"
    workspace.mkdir()
    (workspace / "Cargo.toml").write_text("[workspace]\n")
    monkeypatch.setattr(conftest, "__file__", str(fixture_path))
    monkeypatch.delenv("HOLA_CLI_BINARY", raising=False)
    target = tmp_path / "build" if absolute else Path("build")
    monkeypatch.setenv("CARGO_TARGET_DIR", str(target))
    effective = target if absolute else workspace / target
    binary = _executable(effective / "debug" / ("hola.exe" if os.name == "nt" else "hola"))
    calls = []

    def built(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(conftest.subprocess, "run", built)
    assert conftest.cli_binary.__wrapped__() == str(binary)
    assert calls == [(["cargo", "build", "--locked", "-p", "hola-cli"], workspace)]


def test_required_build_failure_is_not_a_skip(tmp_path, monkeypatch):
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    (workspace / "Cargo.toml").write_text("[workspace]\n")
    monkeypatch.setattr(conftest, "__file__", str(workspace / "hola-py/tests/conftest.py"))
    monkeypatch.delenv("HOLA_CLI_BINARY", raising=False)
    monkeypatch.setattr(
        conftest.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="bad build"),
    )
    with pytest.raises(pytest.fail.Exception, match="Failed to build required hola-cli: bad build"):
        conftest.cli_binary.__wrapped__()


def _noisy_cli(tmp_path, monkeypatch):
    script = tmp_path / "noisy_server.py"
    script.write_text(
        """
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

for stream, byte in ((sys.stdout, b"O"), (sys.stderr, b"E")):
    stream.buffer.write(byte * (1024 * 1024))
    stream.flush()
if Path(sys.argv[1]).stem == "failed":
    print("intentional startup failure after noisy output", file=sys.stderr, flush=True)
    raise SystemExit(7)

class Handler(BaseHTTPRequestHandler):
    count = 0

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        Handler.count += 1
        for stream in (sys.stdout, sys.stderr):
            print("request " + self.path + " " + "L" * 8192, file=stream, flush=True)
        payload = json.dumps({"request_count": Handler.count}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.do_GET()

HTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    real_popen = conftest.subprocess.Popen
    children = []
    logs = []

    def launch(command, **kwargs):
        port = command[command.index("--port") + 1]
        proc = real_popen([sys.executable, "-u", str(script), command[2], port], **kwargs)
        children.append(proc)
        logs.append(kwargs["stdout"])
        return proc

    monkeypatch.setattr(conftest.subprocess, "Popen", launch)
    return children, logs


def test_running_server_serves_after_large_stdout_and_stderr(tmp_path, monkeypatch, free_port):
    """Live server output must not fill an unread pipe and block HTTP responses."""
    children, logs = _noisy_cli(tmp_path, monkeypatch)
    request = SimpleNamespace(
        node=SimpleNamespace(get_closest_marker=lambda name: None),
    )
    fixture = conftest.running_server.__wrapped__("fake-cli", free_port, tmp_path, request)
    try:
        url = next(fixture)
        for index in range(20):
            status, body = conftest.http_json(f"{url}/api/ask", method="POST", timeout=1)
            assert status == 200
            assert body["request_count"] == 2 * index + 2
            status, _ = conftest.http_json(
                f"{url}/api/tell", method="POST", body={"trial_id": index}, timeout=1
            )
            assert status == 200
    finally:
        fixture.close()

    assert len(children) == 1
    assert children[0].returncode is not None
    assert all(log.closed for log in logs)


def test_noisy_startup_failure_keeps_bounded_diagnostics_and_reaps(
    tmp_path, monkeypatch, free_port
):
    children, logs = _noisy_cli(tmp_path, monkeypatch)
    config = tmp_path / "failed.yaml"
    config.write_text("unused fake CLI configuration", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="intentional startup failure") as failure:
        conftest._start_server("fake-cli", config, free_port, attempts=1, startup_timeout=1)

    assert len(str(failure.value)) < 9000
    assert children[0].returncode == 7
    assert all(log.closed for log in logs)


def test_server_stop_waits_after_kill_and_closes_handles():
    calls = []
    streams = [io.BytesIO() for _ in range(3)]

    def wait(*, timeout):
        calls.append("wait")
        if calls.count("wait") == 1:
            raise conftest.subprocess.TimeoutExpired("fixture child", timeout)
        return -9

    proc = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: calls.append("terminate"),
        wait=wait,
        kill=lambda: calls.append("kill"),
        stdin=streams[0],
        stdout=streams[1],
        stderr=streams[2],
    )
    conftest._stop_server(proc)
    assert calls == ["terminate", "wait", "kill", "wait"]
    assert all(stream.closed for stream in streams)
