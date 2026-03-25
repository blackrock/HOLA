# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Smoke tests for the benchmark framework."""

import subprocess
import sys
from pathlib import Path

import pytest

HOLA_PY_DIR = Path(__file__).parent.parent


@pytest.mark.benchmarks
def test_benchmark_imports():
    """All benchmark subpackages import successfully."""
    import benchmarks.adapters.base  # noqa: F401
    import benchmarks.adapters.hola_adapter  # noqa: F401
    import benchmarks.adapters.igr_adapter  # noqa: F401
    import benchmarks.adapters.optuna_adapter  # noqa: F401
    import benchmarks.adapters.pymoo_multi  # noqa: F401
    import benchmarks.adapters.pymoo_single  # noqa: F401
    import benchmarks.data.normalize  # noqa: F401
    import benchmarks.functions.dtlz  # noqa: F401
    import benchmarks.functions.grouped_tlp  # noqa: F401
    import benchmarks.functions.single_objective  # noqa: F401
    import benchmarks.functions.wfg  # noqa: F401
    import benchmarks.functions.zdt  # noqa: F401
    import benchmarks.problems.grouped_tlp  # noqa: F401
    import benchmarks.problems.multi_objective  # noqa: F401
    import benchmarks.problems.single_objective  # noqa: F401
    import benchmarks.runner.executor  # noqa: F401


@pytest.mark.benchmarks
def test_benchmark_cli_help():
    """Benchmark CLI parses --help without error."""
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks", "--help"],
        cwd=str(HOLA_PY_DIR),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"CLI help failed:\n{result.stderr[-2000:]}"
    assert "run-single" in result.stdout


@pytest.mark.benchmarks
def test_benchmark_mini_run(tmp_path):
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
            "5",
            "--problems",
            "forrester_1d",
            "--n-workers",
            "1",
        ],
        cwd=str(HOLA_PY_DIR),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Mini benchmark failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
