# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Smoke tests: each example script completes without error."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
# dashboard_stress_test is an interactive demo that runs forever;
# exclude it from automated smoke tests.
INTERACTIVE = {"dashboard_stress_test.py"}
EXAMPLE_SCRIPTS = sorted(s for s in EXAMPLES_DIR.glob("*.py") if s.name not in INTERACTIVE)


@pytest.mark.examples
@pytest.mark.parametrize(
    "script",
    EXAMPLE_SCRIPTS,
    ids=[s.stem for s in EXAMPLE_SCRIPTS],
)
def test_example_runs(script):
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(EXAMPLES_DIR.parent),  # hola-py/, so benchmarks imports resolve
        capture_output=True,
        text=True,
        timeout=300,  # ml_hyperparameters trains real sklearn models
    )
    assert result.returncode == 0, (
        f"{script.name} failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
