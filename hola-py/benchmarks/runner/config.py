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

"""Run configuration for benchmark execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    output_dir: Path = Path("benchmark_results")
    n_runs: int = 50
    n_workers: int = 0  # 0 = os.cpu_count()
    budgets: list[int] = field(default_factory=lambda: [25, 50, 75, 100, 200, 500, 1000])
    problems: list[str] | None = None  # None = all, or filter by name
    optimizers: list[str] | None = None  # None = all, or filter by name
    resume: bool = True  # Skip already-completed runs

    @property
    def effective_workers(self) -> int:
        if self.n_workers <= 0:
            return os.cpu_count() or 1
        return self.n_workers
