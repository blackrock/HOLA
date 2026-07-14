# Contributing to HOLA

## Development setup

```bash
git clone https://github.com/blackrock/HOLA.git
cd HOLA

# Install Rust toolchain (if not already present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --directory hola-py --locked --dev --group benchmarks
```

## Running tests

```bash
# Rust tests (unit + integration)
# --all-features compiles the server integration tests
cargo test --locked --workspace --all-features

# Build and test Python bindings
uv run --directory hola-py maturin develop
uv run --directory hola-py pytest tests/ -v

# Linting
uv run --directory hola-py ruff check .
uv run --directory hola-py ruff format --check .
uv run --directory hola-py ty check .
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
  `cargo clippy --locked --workspace --all-features -- -D warnings`. We
  enforce a maximum line width of 100 characters (`rustfmt.toml`).
- **Python.** Lint with `uv run --directory hola-py ruff check .`
  and format with `uv run --directory hola-py ruff format --check .`.
  Type-check with `uv run --directory hola-py ty check .`.

## Dashboard

The `dashboard/` directory contains a standalone browser UI with
no build step.

To test locally, have the HOLA server host the dashboard so browser
requests stay on the same origin, then open `http://localhost:8000/`.

```bash
cargo run -p hola-cli -- serve hola-cli/examples/example_study.yaml --dashboard ./dashboard
```

The dashboard uses authenticated `fetch` requests, including an incrementally
parsed SSE response. Cross-origin browser access is disabled by default; when
hosting the UI elsewhere, pass its exact origin with `--cors-origin` (repeat
the flag for multiple origins).

## Pull request guidelines

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Ensure `cargo test --locked --workspace --all-features` and
   `uv run --directory hola-py pytest tests/` pass
4. Run linters before pushing (`cargo clippy`, `cargo fmt`,
   `ruff check`, `ruff format`)
5. Keep PRs focused. One feature or fix per PR.
