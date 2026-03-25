# Contributing to HOLA

## Development setup

```bash
git clone https://github.com/blackrock/HOLA.git
cd HOLA

# Install Rust toolchain (if not already present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --dev
```

## Running tests

```bash
# Rust tests (unit + integration)
# --all-features compiles the server integration tests
cargo test --workspace --all-features

# Build and test Python bindings
cd hola-py && uv run maturin develop && cd ..
hola-py/.venv/bin/python -m pytest hola-py/tests/ -v

# Linting
uv run --project hola-py ruff check .
uv run --project hola-py ruff format --check .
uv run --project hola-py ty check .
```

## Feature flags

The `hola` crate has an optional `server` feature that enables
the Axum REST API. The CLI crate enables it by default. When
running `cargo test` on the workspace, use `--all-features` so
that server integration tests compile.

## Licensing and file headers

We distribute all crates and the Python package under
**Apache-2.0** (see [LICENSE-APACHE](LICENSE-APACHE) at the
repository root). Each source file (Rust, Python, dashboard
JS/CSS/HTML) must begin with the Apache 2.0 copyright and license
notice (BlackRock / 2026). If you add a new file, copy the header
from an existing file of the same type.

## Code style

- **Rust.** Run `cargo fmt --all` before committing. Lint with
  `cargo clippy --workspace --all-features -- -D warnings`. We
  enforce a maximum line width of 100 characters (`rustfmt.toml`).
- **Python.** Lint with `uv run --project hola-py ruff check .`
  and format with `uv run --project hola-py ruff format --check .`.
  Type-check with `uv run --project hola-py ty check .`.

## Dashboard

The `dashboard/` directory contains a standalone browser UI with
no build step.

To test locally, start the server, open `dashboard/index.html` in
a browser, and enter `http://localhost:8000` as the server URL.

```bash
cargo run -p hola-cli -- serve hola-cli/examples/example_study.yaml
```

The dashboard connects via `fetch` and `EventSource` (SSE). CORS
is permissive by default on the server.

## Pull request guidelines

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Ensure `cargo test --workspace --all-features` and
   `hola-py/.venv/bin/python -m pytest hola-py/tests/` pass
4. Run linters before pushing (`cargo clippy`, `cargo fmt`,
   `ruff check`, `ruff format`)
5. Keep PRs focused. One feature or fix per PR.
