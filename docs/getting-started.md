# Getting Started

This guide walks you through installing HOLA, running your first
optimization, and verifying everything works.

## Installation

### Python (pre-built wheel)

We publish pre-built wheels so you do not need a Rust toolchain.
pip automatically selects the right wheel for your platform.

```bash
pip install hola --extra-index-url https://blackrock.github.io/HOLA/simple/
```

We support Python 3.10+ on Linux (x86_64, aarch64), macOS (Intel,
Apple Silicon), and Windows (x86_64).

### CLI binary

Download a pre-built binary from the
[latest release](https://github.com/blackrock/HOLA/releases/latest).
No installation required. Just untar and run.

### From source

Building from source requires the
[Rust toolchain](https://rustup.rs/) (stable), Python 3.10+, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/blackrock/HOLA.git
cd HOLA
```

We build the Python bindings with maturin.

```bash
cd hola-py && uv sync --dev
uv run maturin develop
```

We build the CLI binary with cargo.

```bash
cargo build -p hola-cli --release
# Binary at: target/release/hola
```

## Your First Optimization (Python)

This example minimizes the 1D Forrester function, a standard
benchmark with a known minimum of $-6.0267$ at $x = 0.7572$.

```python
from hola import Study, Space, Real, Minimize

# Define a 1D parameter space
study = Study(
    space=Space(x=Real(0.0, 1.0)),
    objectives=[Minimize("value")],
    strategy="sobol",
    seed=42,  # for reproducibility
)

# Define the objective function
import math

def forrester(params):
    x = params["x"]
    term = 6 * x - 2
    value = term ** 2 * math.sin(term / 2)
    return {"value": value}

# Run 50 trials
study.run(forrester, n_trials=50)

# Inspect the result
best = study.top_k(1)[0]
print(f"Best value: {best.score_vector:.4f}")
print(f"At x = {best.params['x']:.4f}")
```

Here is what each step does.

1. `Space(x=Real(0.0, 1.0))` defines a single parameter `x`
   in $[0, 1]$.
2. `Minimize("value")` tells HOLA to minimize the `"value"` field
   returned by the objective function.
3. `study.run(forrester, n_trials=50)` runs 50 ask/tell iterations
   automatically.
4. `study.top_k(1)` returns a list with the best `CompletedTrial`,
   which has `.score_vector` (scalar score), `.params` (parameter
   values), and `.id`.

## Your First Optimization (CLI)

We start a server with the example study configuration, then run
a worker.

**Terminal 1. Start the server.**

```bash
hola serve hola-cli/examples/example_study.yaml --port 8000
```

**Terminal 2. Run a worker.**

```bash
hola worker --server http://localhost:8000 --exec "python my_training.py"
```

The worker receives trial parameters via the `HOLA_PARAMS`
environment variable (a JSON string) and must print a JSON metrics
object to stdout. See the [CLI Guide](cli-guide.md) for details
on writing worker scripts.

## Verification

If you installed from a pre-built wheel, you are ready to go. The
test suites below are for contributors who build from source.

We run the Rust tests with

```bash
cargo test --workspace --all-features
```

We run the Python tests directly through the venv python (not
`uv run`) so that the maturin-built extension is picked up.

```bash
hola-py/.venv/bin/python -m pytest hola-py/tests/ -v
```

We lint with

```bash
cargo clippy --workspace --all-features -- -D warnings
uv run ruff check .
```

## Running Examples

We run all examples from the repository root so that paths resolve
correctly.

```bash
# Python examples (run as scripts)
uv run python hola-py/examples/basic_optimization.py
uv run python hola-py/examples/categorical_demo.py
uv run python hola-py/examples/multi_objective.py

# Rust examples
cargo run -p opt_engine --example rastrigin_gmm
cargo run -p opt_engine --example pareto_front
cargo run -p opt_engine --example persistence
```

If you use a `uv` project rooted in `hola-py/`, run the same
paths relative to that directory, e.g.,
`uv run python examples/basic_optimization.py`.

## What's Next

- [Python Guide](python-guide.md). Full API reference for spaces,
  objectives, strategies, and `Study`.
- [CLI & Distributed](cli-guide.md). YAML config format and
  multi-machine setup.
- [Concepts](concepts.md). How strategies, scalarization, and the
  optimization pipeline work.
