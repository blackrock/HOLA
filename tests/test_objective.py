# Copyright 2021 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from hola.objective import ObjectiveConfig, ObjectiveScalarizer


def test_parse() -> None:
    assert ObjectiveConfig(target=1, limit=2, priority=3) == ObjectiveConfig.parse(
        {"target": 1, "limit": 2, "priority": 3}
    )


def test_scalarize() -> None:
    config = ObjectiveConfig(target=1, limit=2)
    scalarizer = ObjectiveScalarizer({"f": config})
    assert scalarizer.scalarize_objectives({"f": 1.5}) == 0.5
    assert scalarizer.scalarize_objectives({"f": 3}) == np.inf
    assert scalarizer.scalarize_objectives({"f": -100}) == 0
