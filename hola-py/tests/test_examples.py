# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Smoke tests: each example script completes without error."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
BENCHMARKS_DIR = EXAMPLES_DIR.parent / "benchmarks"
# dashboard_stress_test is an interactive demo that runs forever;
# exclude it from automated smoke tests.
INTERACTIVE = {"dashboard_stress_test.py"}
EXAMPLE_SCRIPTS = sorted(s for s in EXAMPLES_DIR.glob("*.py") if s.name not in INTERACTIVE)


@pytest.fixture(scope="session")
def isolated_benchmarks_path(tmp_path_factory):
    """Expose benchmark helpers without adding the Python source tree.

    Example subprocesses must import ``hola_opt`` from the installed wheel.
    Copying only ``benchmarks`` into a temporary import root keeps the examples'
    repository-only helpers available without making the adjacent source
    package importable. A copy is portable to Windows, where creating symlinks
    may require additional privileges.
    """
    import_root = tmp_path_factory.mktemp("example-imports")
    shutil.copytree(
        BENCHMARKS_DIR,
        import_root / "benchmarks",
        ignore=shutil.ignore_patterns("__pycache__", "*.py[co]"),
    )
    return import_root


@pytest.mark.examples
@pytest.mark.parametrize(
    "script",
    EXAMPLE_SCRIPTS,
    ids=[s.stem for s in EXAMPLE_SCRIPTS],
)
def test_example_runs(script, isolated_benchmarks_path):
    env = os.environ.copy()
    # Deliberately replace, rather than extend, an inherited PYTHONPATH. The
    # hola-py source root must not shadow the exact wheel installed by CI.
    env["PYTHONPATH"] = str(isolated_benchmarks_path)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(isolated_benchmarks_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,  # ml_hyperparameters trains real sklearn models
    )
    assert result.returncode == 0, (
        f"{script.name} failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
