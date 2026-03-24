# HOLA

**Hyperparameter Optimization, Lightweight Asynchronous.** A Python
library for black-box optimization, backed by a Rust engine for
speed.

We provide a simple **ask/tell** interface that works the same way
whether the engine runs in your Python process or on a remote
server. For **local** runs, `Study` embeds the engine in-process
with no network overhead. For **distributed** deployments, the
study server speaks REST and SSE so workers, clients, and the
dashboard can connect from any language or machine.

## Features

**Ask/tell interface.**
:   One API across Python, CLI, and REST. `study.ask()` proposes
    a trial; `study.tell()` returns the result.

**Multiple strategies.**
:   Sobol sequences, random search, and Gaussian mixture model
    (GMM) with automatic refitting.

**Multi-objective optimization.**
:   Pareto fronts, scalarized ranking, and target-limit-priority
    (TLP) objectives with priority groups.

**Flexible parameter spaces.**
:   Continuous, integer, and categorical parameters with linear,
    log, and log10 scales.

**Real-time dashboard.**
:   Live convergence plots, Pareto scatter, parallel coordinates,
    and a sortable trial table.

**Persistence.**
:   Atomic JSON checkpoints that capture the full engine
    state---leaderboard, strategy, and configuration.

**Distributed execution.**
:   REST API with SSE streaming for real-time updates across
    machines.

## Quick Start

```bash
pip install hola --extra-index-url https://blackrock.github.io/HOLA/simple/
```

```python
from hola import Study, Space, Real, Integer, Categorical, Minimize

study = Study(
    space=Space(
        lr=Real(1e-4, 0.1, scale="log10"),
        layers=Integer(1, 10),
        optimizer=Categorical(["adam", "sgd", "rmsprop"]),
    ),
    objectives=[Minimize("loss")],
    strategy="sobol",
)

def objective(params):
    return {"loss": train(params)}

study.run(objective, n_trials=100)
print(study.top_k(1))
```

See the [Getting Started](getting-started.md) guide for
installation instructions and a detailed walkthrough.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Installation, first optimization, verification |
| [Python Guide](python-guide.md) | Full Python API: spaces, objectives, strategies, `Study`, `Study.connect()` |
| [CLI & Distributed](cli-guide.md) | YAML config, `hola serve`, `hola worker`, multi-machine setup |
| [REST API](rest-api.md) | Endpoint reference with request/response schemas |
| [Concepts](concepts.md) | Architecture, strategies, scalarization, the unit hypercube |
| [Dashboard](dashboard.md) | Real-time visualization and checkpoint analysis |

## Upgrading from the Python-only release

*Skip this section if you are new to HOLA.* It applies only if you
used the older Python-first HOLA (`hola.tune`, `hola_serve`,
`/report_request`, and related APIs). That API lived in
[earlier commits](https://github.com/blackrock/HOLA/commits/) on
this repository. This codebase is a **rewrite** with a Rust core,
a new Python surface (`Study` / `Study.connect()`), and a new REST
protocol (see [REST API](rest-api.md)).

| Topic | Old Python release | This release |
|-------|-------------------------|-----------------------------|
| Core runtime | Python | Rust with Python bindings (PyO3) |
| Python API | `tune()`, leaderboard CSV | `Study`, `Study.connect()`, ask/tell |
| Server | `hola_serve`, `/report_request`, ... | Axum REST + SSE (see [REST API](rest-api.md)) |
| Config | JSON files in a directory | YAML study config + CLI (`hola serve`) |
| Worker env | varies | `HOLA_PARAMS` JSON via `hola worker` |

There is **no drop-in replacement** for the old Python API or HTTP
routes. We recommend migrating to the ask/tell interface, the
YAML/CLI server, and the REST API documented in these guides.

## License

Licensed under Apache 2.0. See `LICENSE-APACHE` in the repository
root.
