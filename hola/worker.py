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

from typing import cast

import requests

from hola.objective import FuncName
from hola.params import ParamName


class Worker:
    def __init__(self, server_url: str = "http://localhost", port: int = 8675):
        self.url = server_url
        self.port = port
        self.rep_req_url = f"{self.url}:{self.port}/report_request"

    def get_param_sample(self) -> dict:
        new_sample = requests.get(self.rep_req_url).json()
        return cast(dict, new_sample)

    def report_sim_result(
        self, objectives: dict[FuncName, float] | None = None, params: dict[ParamName, float] | None = None
    ) -> dict:
        sim_result = {}
        if objectives is not None and params is None:
            sim_result = {"params": params, "objectives": objectives}
        new_sample = requests.post(self.rep_req_url, json=sim_result).json()
        return cast(dict, new_sample)
