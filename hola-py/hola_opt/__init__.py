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

import pathlib

from .hola_opt import *  # noqa: F403
from .hola_opt import (
    Categorical,
    CheckpointError,
    CompletedTrial,
    ConfigurationError,
    Gmm,
    HolaError,
    Integer,
    Maximize,
    Minimize,
    ObjectiveError,
    Random,
    Real,
    RemoteError,
    Sobol,
    Space,
    Study,
    Trial,
)

__all__ = [
    "HolaError",
    "ConfigurationError",
    "CheckpointError",
    "RemoteError",
    "ObjectiveError",
    "Real",
    "Integer",
    "Categorical",
    "Minimize",
    "Maximize",
    "Gmm",
    "Sobol",
    "Random",
    "Space",
    "Study",
    "Trial",
    "CompletedTrial",
    "dashboard_dir",
]


def dashboard_dir() -> pathlib.Path:
    """Return the path to the bundled dashboard static files.

    Useful for passing to ``Study.serve(dashboard_path=str(dashboard_dir()))``.
    The same versioned assets are included in editable source trees, wheels, and
    source distributions.
    """
    return pathlib.Path(__file__).parent / "dashboard"
