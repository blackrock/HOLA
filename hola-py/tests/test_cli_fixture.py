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
import os
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
