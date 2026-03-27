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

"""Export utilities for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure in PDF, PNG, and PGF formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / f"{name}.pgf", bbox_inches="tight")
