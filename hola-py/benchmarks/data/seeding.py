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

"""Canonical deterministic seed derivation for benchmark execution and audit."""

from __future__ import annotations

import hashlib


def make_seed(problem_name: str, budget: int, run_id: int) -> int:
    """Return the paired optimizer seed for one problem, horizon, and run."""
    material = f"{problem_name}:{budget}:{run_id}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def make_hpo_split_seed(problem_name: str, run_id: int) -> int:
    """Return the HPO split seed fixed across optimizers and search horizons."""
    material = f"hpo-split:{problem_name}:{run_id}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16)
