# Rust Library

!!! warning "Internal API. No stability guarantees."

    The `opt_engine` crate is an internal implementation detail
    of HOLA. Its public surface may change without notice between
    releases. We recommend the [Python API](python-guide.md) for
    all optimization workflows.

## Architecture

We organize the Rust internals into three layers. The **engine
core** (`opt_engine`) provides generic, type-safe optimization
building blocks: spaces, strategies, scales, and the leaderboard.
The **orchestration layer** (`hola`) composes those blocks into the
optimization loop behind a
JSON-based, type-erased interface that resolves spaces and
strategies at runtime from configuration. The **public
interfaces** (Python bindings, the CLI, and the REST server) all
delegate to the orchestration layer.

## Exploring the source

If you want to browse the internal API documentation, run

```bash
cargo doc -p opt_engine --open
```

The inline doc comments serve as the primary
reference for Rust-level details.
