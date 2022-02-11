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

from hola.params import ParamConfig, ParameterTransformer


def test_parse() -> None:
    assert ParamConfig(min=1, max=2) == ParamConfig.parse({"min": 1, "max": 2})


def test_param_transform() -> None:
    transformer = ParameterTransformer({"x1": ParamConfig(min=1, max=2)})
    assert transformer.transform_u_params(np.array([1])) == {"x1": 2.0}
    assert np.array_equal(transformer.back_transform_param_dict({"x1": 2}), np.array([1]))
