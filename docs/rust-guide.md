# Rust Library

!!! warning "Internal API. No stability guarantees."

    The `opt_engine` crate is an internal implementation detail
    of HOLA. Its public surface may change without notice between
    releases. We recommend the [Python API](python-guide.md) for
    all optimization workflows.

## Architecture

We organize the Rust internals into three layers. The **engine
core** (`opt_engine`) provides a generic, type-safe optimization
loop parameterized over space, strategy, and transformer. The
**orchestration layer** (`hola`) wraps that core behind a
JSON-based, type-erased interface that resolves spaces and
strategies at runtime from configuration. The **public
interfaces** (Python bindings, the CLI, and the REST server) all
delegate to the orchestration layer.

## Exploring the source

If you want to browse the internal API documentation, run

```bash
cargo doc -p opt_engine --open
```

The inline doc comments are comprehensive and serve as the primary
reference for Rust-level details.
