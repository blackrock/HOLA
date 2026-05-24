# Dependency Feature Sets Are Broader Than Needed

## Status

Resolved in branch `audit/known-issues` with the issue 10 fix.

## Tags

- severity: low
- type: maintenance
- area: dependencies
- area: build
- area: security-surface
- regression-risk: low

## Summary

The Rust crates use broad dependency feature sets, notably `tokio` with
`features = ["full"]`. This increases compile time, dependency graph size, and
security/update surface beyond what the code appears to need.

## Evidence

- `opt_engine` depends on Tokio with all features.
  - `opt_engine/Cargo.toml:10`
- `hola` depends on Tokio with all features.
  - `hola/Cargo.toml:15`
- `hola-cli` depends on Tokio with all features.
  - `hola-cli/Cargo.toml:17`

## Impact

- Slower clean builds and CI.
- More transitive code included in the dependency graph.
- Larger audit and vulnerability review surface.

## Proposed Fix

Minimize features per crate.

Likely needed features:

- `opt_engine`: `sync`, `rt`, and possibly `rt-multi-thread` only if tests or
  `spawn_blocking` require it.
- `hola`: `sync`, `rt`, `rt-multi-thread`, `net`, `macros` depending on server
  and tests.
- `hola-cli`: `macros`, `rt-multi-thread`, `time`, `process` if async process
  APIs are used, otherwise standard process does not need Tokio process.

Validate by reducing features incrementally and running the full Rust and
Python test suites.

## Resolution

- Replaced `tokio = { features = ["full"] }` in `opt_engine`, `hola`,
  `hola-cli`, and `hola-py` with explicit minimal feature sets.
- `opt_engine` now enables only `macros`, `rt-multi-thread`, and `sync` for
  async locks, blocking refit tasks, and tests/examples.
- `hola` now enables only `macros`, `rt-multi-thread`, and `sync` by default;
  the `server` feature adds `tokio/net` and the minimal Axum features needed
  for JSON, query extractors, HTTP/1, and Tokio serving.
- `hola-cli` now enables only `macros`, `rt-multi-thread`, and `time` for the
  Tokio main runtime, async HTTP, and retry sleeps.
- `hola-py` now enables only `rt-multi-thread` for constructing runtimes from
  Python bindings.
- Disabled unnecessary default features on optional server dependencies where
  the code uses narrower feature sets: `axum`, `tokio-stream`, and
  `tower-http`.
- Updated `Cargo.lock`; the resolved graph no longer includes Tokio
  full-feature-only dependencies such as `parking_lot` and
  `signal-hook-registry`.

## Acceptance Criteria

- Replace `tokio = { features = ["full"] }` with the minimal feature set in
  each crate.
- `cargo check --workspace --all-features` passes.
- `cargo test --workspace --all-features` passes.
- Python test suite still passes, especially server and CLI integration tests.
- Document why each enabled feature is needed if the feature list is not
  obvious.

## Suggested Tests

- Run `cargo tree -e features` before and after to confirm feature reduction.
- Run the full existing verification suite.

## Verification

- `cargo tree -e features -i tokio --workspace --all-features`
- `cargo check --workspace --all-features`
- `cargo check --workspace`
- `cargo test --workspace --all-features`
- `cargo check --workspace --all-features --all-targets`
- `cd hola-py && uv run maturin develop`
- `cd hola-py && uv run pytest -q`

All passed after the fix.
