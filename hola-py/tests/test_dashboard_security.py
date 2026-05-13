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

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dashboard_app_avoids_html_sinks_for_untrusted_data():
    app_js = (ROOT / "dashboard" / "app.js").read_text()

    forbidden = [
        "innerHTML",
        "insertAdjacentHTML",
        "outerHTML",
        "document.write",
    ]
    for token in forbidden:
        assert token not in app_js


def test_dashboard_xss_smoke_fixture_contains_html_like_values():
    fixture = json.loads((ROOT / "dashboard" / "xss-smoke-checkpoint.json").read_text())
    trial = fixture["leaderboard"]["trials"][0]

    joined = json.dumps(trial)
    assert "<img src=x onerror=alert(1)>" in joined
    assert "<script>alert(4)</script>" in joined
