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

"""CLI entry point: python -m benchmarks <command>."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="benchmarks",
        description="HOLA benchmark suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run-single", help="Run single-objective benchmarks")
    subparsers.add_parser("run-multi", help="Run multi-objective benchmarks")
    subparsers.add_parser("run-hpo", help="Run the practical mixed-space HPO benchmark")
    subparsers.add_parser("run-grouped-tlp", help="Run the grouped-TLP capability benchmark")
    subparsers.add_parser("plot-single", help="Generate single-objective plots")
    subparsers.add_parser("plot-multi", help="Generate multi-objective plots")
    subparsers.add_parser("plot-hpo", help="Report the practical HPO campaign")
    subparsers.add_parser("plot-grouped-tlp", help="Report the grouped-TLP campaign")

    args, remaining = parser.parse_known_args()

    # Re-parse with the subcommand's own argument parser
    sys.argv = [f"benchmarks {args.command}"] + remaining

    if args.command == "run-single":
        from benchmarks.runner.run_single_objective import main as run_so

        run_so()
    elif args.command == "run-multi":
        from benchmarks.runner.run_multi_objective import main as run_mo

        run_mo()
    elif args.command == "run-hpo":
        from benchmarks.runner.run_hpo import main as run_hpo

        run_hpo()
    elif args.command == "run-grouped-tlp":
        from benchmarks.runner.run_grouped_tlp import main as run_grouped_tlp

        run_grouped_tlp()
    elif args.command == "plot-single":
        from benchmarks.plotting.single_objective import main as plot_so

        plot_so()
    elif args.command == "plot-multi":
        from benchmarks.plotting.multi_objective import main as plot_mo

        plot_mo()
    elif args.command == "plot-hpo":
        from benchmarks.plotting.hpo import main as plot_hpo

        plot_hpo()
    elif args.command == "plot-grouped-tlp":
        from benchmarks.plotting.grouped_tlp import main as plot_grouped_tlp

        plot_grouped_tlp()


if __name__ == "__main__":
    main()
