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
from __future__ import annotations

from typing import MutableMapping, Type, TypeVar

from pydantic import BaseModel

_Model = TypeVar("_Model", bound=BaseModel)


def parse(cls: Type[_Model], param_dict: _Model | MutableMapping | str) -> _Model:
    """Parse object of type cls, dictionary or json string into object of type cls."""
    if isinstance(param_dict, cls):
        return param_dict
    if isinstance(param_dict, str):
        return cls.parse_raw(param_dict)
    if isinstance(param_dict, MutableMapping):
        return cls.parse_obj(param_dict)
    raise TypeError(f"Only {cls.__name__}, str and dict can be parsed into a {cls.__name__}")
