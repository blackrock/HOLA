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

"""Public exception hierarchy and error-mapping regressions."""

import socket
import threading
from contextlib import suppress

import pytest

from hola_opt import (
    CheckpointError,
    ConfigurationError,
    HolaError,
    Minimize,
    ObjectiveError,
    Real,
    RemoteError,
    Space,
    Study,
)


def _study():
    return Study(space=Space(x=Real(0.0, 1.0)), objectives=[Minimize("loss")])


def test_exception_hierarchy_preserves_value_error_compatibility():
    for error_type in (ConfigurationError, CheckpointError, RemoteError, ObjectiveError):
        assert issubclass(error_type, HolaError)
        assert issubclass(error_type, ValueError)
    assert issubclass(HolaError, ValueError)


def test_invalid_configuration_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="scale must be") as raised:
        Real(0.0, 1.0, scale="log2")
    assert isinstance(raised.value, ValueError)

    with pytest.raises(ConfigurationError, match="Parameter 'x'"):
        Study(space=Space(x=Real(2.0, 1.0)), objectives=[Minimize("loss")])


def test_checkpoint_load_failure_raises_checkpoint_error(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(CheckpointError, match="Failed to load checkpoint"):
        Study.load(str(missing))


def test_remote_transport_and_url_errors_are_distinct():
    with pytest.raises(ConfigurationError, match="Invalid server URL"):
        Study.connect("not-a-url")

    # An active listener avoids OS-specific closed-port behavior (Windows can
    # silently drop loopback SYNs). Sending plaintext to an HTTPS client forces
    # a deterministic connector-stage TLS failure on every supported platform.
    server_errors = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2.0)
        host, port = listener.getsockname()

        def serve_plaintext_once():
            try:
                connection, _ = listener.accept()
                with connection, suppress(BrokenPipeError, ConnectionResetError):
                    connection.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            except OSError as error:
                server_errors.append(error)

        thread = threading.Thread(target=serve_plaintext_once, daemon=True)
        thread.start()
        try:
            remote = Study.connect(
                f"https://{host}:{port}",
                connect_timeout=1.0,
                request_timeout=2.0,
            )
            with pytest.raises(RemoteError, match="HTTP connection failed") as raised:
                remote.ask()
        finally:
            thread.join(timeout=3.0)

        assert not thread.is_alive(), "plaintext TLS test server did not finish"
        assert not server_errors, f"plaintext TLS test server failed: {server_errors}"
    assert isinstance(raised.value, ValueError)


def test_invalid_objective_result_raises_objective_error():
    study = _study()
    with pytest.raises(ObjectiveError, match="return a dict"):
        study.run(lambda _params: 42, n_trials=1)

    trial = study.ask()
    with pytest.raises(ObjectiveError, match="Cannot convert Python object to JSON"):
        study.tell(trial.trial_id, {"loss": object()})


def test_user_objective_exception_is_propagated_unchanged():
    class UserObjectiveFailure(RuntimeError):
        pass

    def objective(_params):
        raise UserObjectiveFailure("training exploded")

    with pytest.raises(UserObjectiveFailure, match="training exploded"):
        _study().run(objective, n_trials=1)
