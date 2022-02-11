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
import requests

SERVER_ADDR = "http://localhost:8675"

resp = requests.get(SERVER_ADDR + "/report_request")

print(resp.json())

resp = requests.post(SERVER_ADDR + "/report_request", json={})

print(resp.json())

params = resp.json()

result = {"r_squared": 1, "abs_error": 0}

data = {"params": params, "objectives": result}
resp = requests.post(SERVER_ADDR + "/report_request", json=data)

print(resp.json())
