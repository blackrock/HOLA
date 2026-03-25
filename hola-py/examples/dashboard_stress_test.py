# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Dashboard stress test.

Runs a multi-objective optimization with adversarial metrics and serves the
dashboard so you can observe how it handles edge cases in real time.

Usage
-----
    cd hola-py && uv run maturin develop && cd ..
    hola-py/.venv/bin/python hola-py/examples/dashboard_stress_test.py

Then open http://localhost:8000 in a browser.

After the run completes, the script saves a checkpoint to
``dashboard_checkpoint.json`` for offline testing. You can load this file in
the dashboard via the "Load File" button.
"""

from __future__ import annotations

import math
import pathlib
import random
import time

from hola_opt import (
    Categorical,
    Gmm,
    Integer,
    Maximize,
    Minimize,
    Real,
    Space,
    Study,
    dashboard_dir,
)

N_TRIALS = 200


def adversarial_objective(params: dict) -> dict:
    """Produce metrics with occasional extreme values."""
    x = params["x"]
    y = params["y"]
    n = params["n"]

    base_loss = x**2 + math.log10(max(y, 1e-12))

    # Occasionally produce extreme values.
    r = random.random()
    if r < 0.03:
        base_loss = float("inf")
    elif r < 0.06:
        base_loss = 1e15

    throughput = max(1.0, 1000.0 - abs(x) * 100 + n)
    memory = y * n

    return {"loss": base_loss, "throughput": throughput, "memory": memory}


def main() -> None:
    study = Study(
        space=Space(
            x=Real(-10.0, 10.0),
            y=Real(1e-6, 1e6, scale="log10"),
            n=Integer(0, 1000),
            algo=Categorical(["a", "b", "c", "d", "e", "f", "g", "h"]),
        ),
        objectives=[
            Minimize("loss", target=0.0, limit=100.0, priority=1.0, group="quality"),
            Maximize("throughput", target=1000.0, limit=1.0, priority=0.5, group="perf"),
            Minimize("memory", target=0.0, limit=1e9, priority=0.3, group="perf"),
        ],
        strategy=Gmm(refit_interval=10, elite_fraction=0.15),
        seed=42,
    )

    # Serve the dashboard in the background.
    # dashboard_dir() points to the bundled dashboard in release wheels.
    # For editable installs from source, we fall back to the repo root.
    dash_path = dashboard_dir()
    if not dash_path.exists():
        dash_path = pathlib.Path(__file__).resolve().parents[2] / "dashboard"
    print(f"Serving dashboard from {dash_path}")
    print("Open http://localhost:8000 in a browser to watch.")
    study.serve(port=8000, background=True, dashboard_path=str(dash_path))

    # Give the server a moment to start.
    time.sleep(0.5)

    # Run trials one at a time so the dashboard can show live updates.
    for i in range(N_TRIALS):
        trial = study.ask()
        metrics = adversarial_objective(trial.params)
        study.tell(trial.trial_id, metrics)
        if (i + 1) % 50 == 0:
            best = study.top_k(1)
            score = best[0].score_vector if best else "N/A"
            print(f"Trial {i + 1}/{N_TRIALS} — best score: {score}")
        # Small delay so the dashboard SSE stream can keep up visually.
        time.sleep(0.05)

    # Save checkpoint for offline testing via the dashboard's "Load File" button.
    study.save("dashboard_checkpoint.json")
    print(f"\nDone. {N_TRIALS} trials completed.")
    print("Checkpoint saved to dashboard_checkpoint.json")
    print("You can load this file in the dashboard for offline analysis.")
    print("Press Ctrl+C to stop the server.")

    # Keep the server alive for manual inspection.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
