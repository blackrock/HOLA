# HOLA

**Hyperparameter Optimization, Lightweight Asynchronous.** A Python
library for black-box optimization, backed by a Rust engine for speed.

We provide a simple **ask/tell** interface that works the same way
whether the engine runs in your Python process or on a remote server.
Define a parameter space, choose an objective, and let HOLA suggest
trials.

## Installation

Pre-built wheels are available for Linux, macOS, and Windows
(Python 3.10+).

```bash
pip install hola-opt --extra-index-url https://blackrock.github.io/HOLA/simple/
```

To build from source instead, see the
[Getting Started](docs/getting-started.md) guide.

## Quick Start

This example minimizes the one-dimensional Forrester function, a
standard benchmark with a known minimum of approximately -6.03 near
*x* = 0.757.

```python
from hola_opt import Study, Space, Real, Minimize
import math

study = Study(
    space=Space(x=Real(0.0, 1.0)),
    objectives=[Minimize("value")],
    strategy="sobol",
    seed=42,
)

def forrester(params):
    x = params["x"]
    term = 6 * x - 2
    return {"value": term ** 2 * math.sin(term / 2)}

study.run(forrester, n_trials=50)

best = study.top_k(1)[0]
print(f"Best value: {best.score_vector}")
print(f"At x = {best.params['x']:.4f}")
```

`Space(x=Real(0.0, 1.0))` defines a single continuous parameter.
`Minimize("value")` tells HOLA to minimize the `"value"` field
returned by the objective function.
`study.run(forrester, n_trials=50)` automates 50 ask/tell iterations.
`study.top_k(1)` returns the best `CompletedTrial`, which carries
`.score_vector`, `.params`, and `.metrics`.

## Going Distributed

For multi-machine or language-agnostic deployments, HOLA provides a
CLI server and worker. Start a server from a YAML study configuration,
then point workers at it.

```bash
# Terminal 1: start the server
hola serve config.yaml --port 8000

# Terminal 2: run a worker
hola worker --server http://localhost:8000 --exec "python train.py"
```

The worker sets `HOLA_SERVER`, `HOLA_TRIAL_ID`, and `HOLA_PARAMS`
environment variables, then runs your command. Your script reads its
parameters from `HOLA_PARAMS` and calls `POST /api/tell` on the
server to report results. See the [CLI guide](docs/cli-guide.md)
for details.

From Python, `Study.connect()` speaks the same REST protocol without
the CLI.

```python
study = Study.connect("http://localhost:8000")
trial = study.ask()
study.tell(trial.trial_id, {"loss": 0.42})
```

## Dashboard

The `dashboard/` directory contains a zero-install browser UI for
monitoring live studies or exploring saved checkpoints. Open
`dashboard/index.html`, enter a server URL, and see convergence
plots, Pareto scatter, parallel coordinates, and a sortable trial
table, all updated in real time via SSE.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first optimization, verification |
| [Python Guide](docs/python-guide.md) | Full Python API: spaces, objectives, strategies, `Study`, `Study.connect()` |
| [CLI & Distributed](docs/cli-guide.md) | YAML config, `hola serve`, `hola worker`, multi-machine setup |
| [REST API](docs/rest-api.md) | Endpoint reference with request/response schemas |
| [Concepts](docs/concepts.md) | Architecture, strategies, scalarization, the unit hypercube |
| [Dashboard](docs/dashboard.md) | Real-time visualization and checkpoint analysis |

## Development

```bash
# Run all Rust tests
cargo test --workspace --all-features

# Build and test Python bindings
cd hola-py && uv sync --dev && uv run maturin develop && cd ..
hola-py/.venv/bin/python -m pytest hola-py/tests/ -v

# Lint
cargo clippy --workspace --all-features -- -D warnings
uv run --project hola-py ruff check .
```

## License

Licensed under [Apache 2.0](LICENSE-APACHE).
