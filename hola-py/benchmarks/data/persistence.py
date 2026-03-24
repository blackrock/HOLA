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

"""CSV-based result persistence with resume support."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.data.schema import MO_COLUMNS, SO_COLUMNS


class ResultStore:
    """Thread-safe CSV result store with resume support."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def so_path(self) -> Path:
        return self.output_dir / "single_objective.csv"

    @property
    def mo_path(self) -> Path:
        return self.output_dir / "multi_objective.csv"

    def append_single(self, row: dict[str, Any]) -> None:
        """Append a single-objective result row."""
        # Serialize convergence trace as JSON
        if "convergence_trace" in row and isinstance(row["convergence_trace"], list):
            row = {**row, "convergence_trace": json.dumps(row["convergence_trace"])}
        self._append_row(self.so_path, SO_COLUMNS, row)

    def append_multi(self, row: dict[str, Any]) -> None:
        """Append a multi-objective result row."""
        self._append_row(self.mo_path, MO_COLUMNS, row)

    def _append_row(self, path: Path, columns: list[str], row: dict[str, Any]) -> None:
        with self._lock:
            write_header = not path.exists()
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    def load_single(self) -> pd.DataFrame:
        """Load all single-objective results."""
        if not self.so_path.exists():
            return pd.DataFrame(columns=SO_COLUMNS)
        return pd.read_csv(self.so_path)

    def load_multi(self) -> pd.DataFrame:
        """Load all multi-objective results."""
        if not self.mo_path.exists():
            return pd.DataFrame(columns=MO_COLUMNS)
        return pd.read_csv(self.mo_path)

    def completed_so_runs(self) -> set[tuple[str, str, int, int]]:
        """Return set of (problem, optimizer, budget, run_id) already completed."""
        df = self.load_single()
        if df.empty:
            return set()
        return set(
            df[["problem", "optimizer", "budget", "run_id"]].itertuples(index=False, name=None)
        )

    def completed_mo_runs(self) -> set[tuple[str, str, int, int]]:
        """Return set of (problem, optimizer, budget, run_id) already completed."""
        df = self.load_multi()
        if df.empty:
            return set()
        return set(
            df[["problem", "optimizer", "budget", "run_id"]].itertuples(index=False, name=None)
        )
