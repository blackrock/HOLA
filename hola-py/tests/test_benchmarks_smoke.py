# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Smoke tests for the benchmark framework."""

import subprocess
import sys

import pytest

BENCHMARK_MODULES = (
    "benchmarks.adapters.base",
    "benchmarks.adapters.hola_adapter",
    "benchmarks.adapters.hpo",
    "benchmarks.adapters.igr_adapter",
    "benchmarks.adapters.optuna_adapter",
    "benchmarks.adapters.pymoo_common",
    "benchmarks.adapters.pymoo_multi",
    "benchmarks.adapters.pymoo_single",
    "benchmarks.data.manifest",
    "benchmarks.data.normalize",
    "benchmarks.data.persistence",
    "benchmarks.functions.dtlz",
    "benchmarks.functions.grouped_tlp",
    "benchmarks.functions.hpo",
    "benchmarks.functions.single_objective",
    "benchmarks.functions.wfg",
    "benchmarks.functions.zdt",
    "benchmarks.problems.grouped_tlp",
    "benchmarks.problems.hpo",
    "benchmarks.problems.multi_objective",
    "benchmarks.problems.single_objective",
    "benchmarks.plotting.bootstrap",
    "benchmarks.plotting.grouped_tlp",
    "benchmarks.plotting.hpo",
    "benchmarks.runner.executor",
    "benchmarks.runner.run_grouped_tlp",
    "benchmarks.runner.run_hpo",
)


@pytest.mark.benchmarks
def test_benchmark_imports(isolated_benchmarks_path, isolated_benchmarks_env):
    """All benchmark subpackages import successfully."""
    result = subprocess.run(
        [sys.executable, "-c", ";".join(f"import {module}" for module in BENCHMARK_MODULES)],
        cwd=str(isolated_benchmarks_path),
        env=isolated_benchmarks_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Benchmark imports failed:\n{result.stderr[-2000:]}"


@pytest.mark.benchmarks
def test_benchmark_cli_help(isolated_benchmarks_path, isolated_benchmarks_env):
    """Benchmark CLI parses --help without error."""
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks", "--help"],
        cwd=str(isolated_benchmarks_path),
        env=isolated_benchmarks_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"CLI help failed:\n{result.stderr[-2000:]}"
    assert "run-single" in result.stdout
    assert "run-hpo" in result.stdout
    assert "run-grouped-tlp" in result.stdout
    assert "plot-hpo" in result.stdout
    assert "plot-grouped-tlp" in result.stdout


@pytest.mark.benchmarks
def test_benchmark_mini_run(tmp_path, isolated_benchmarks_path, isolated_benchmarks_env):
    """Run a minimal single-objective benchmark to verify pipeline."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks",
            "run-single",
            "--output-dir",
            str(tmp_path),
            "--n-runs",
            "1",
            "--budgets",
            "6",
            "--problems",
            "forrester_1d",
            "--n-workers",
            "1",
        ],
        cwd=str(isolated_benchmarks_path),
        env=isolated_benchmarks_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Mini benchmark failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
