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

"""Column names and schemas for benchmark result data."""

# Single-objective result columns
SO_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "run_id",
    "seed",
    "best_value",
    "wall_time_seconds",
    "convergence_trace",
]

# Multi-objective result columns
MO_COLUMNS = [
    "problem",
    "optimizer",
    "budget",
    "run_id",
    "seed",
    "hypervolume",
    "igd",
    "spacing",
    "wall_time_seconds",
    "n_pareto_points",
]
