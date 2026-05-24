# AutoStrategy Exploration Accounting Ignores Pending Trials

## Status

Resolved in branch `audit/known-issues` with the issue 08 fix.

## Tags

- severity: medium
- type: algorithm
- area: strategy
- area: distributed
- area: ask-tell
- user-impact: sampling-quality

## Summary

`AutoStrategy` switches from Sobol exploration to GMM exploitation based on
completed trial count. In distributed usage, many `ask()` calls can be issued
before any `tell()` completes. Because `trial_count` increments only during
`update()`, a batch of pending trials can exceed the exploration budget while
still being sampled from Sobol.

## Evidence

- `suggest()` checks `s.trial_count < s.exploration_budget`.
  - `hola/src/hola_engine.rs:451`
- `trial_count` increments only on `update()`, which occurs during `tell()`.
  - `hola/src/hola_engine.rs:460`
  - `hola/src/hola_engine.rs:466`

## Impact

- Distributed workers can oversample the exploration phase.
- The intended exploration/exploitation split depends on worker concurrency and
  result latency.
- GMM may start later than intended in high-parallelism runs.

## Proposed Fix

Track issued suggestions separately from completed updates.

Recommended design options:

1. Add an issued counter to `AutoStrategy` and increment it in `suggest()`.
2. Or let `HolaEngine::ask()` decide exploration mode based on
   completed + pending counts.

Preferred path:

- Add `issued_count` to `AutoStrategy`.
- Serialize it in checkpoints.
- Use `issued_count` for the Sobol/GMM switch.
- Keep `completed_count` or existing `trial_count` for refit/update logic.

## Resolution

- Added an `issued_count` counter to `AutoStrategy` and increment it from
  `suggest()`, so the Sobol/GMM boundary is based on issued trials rather than
  completed `tell()` calls.
- Kept the existing `trial_count` as completed-update accounting for refit and
  strategy update behavior.
- Serialized `issued_count` in full checkpoints, with backward-compatible
  deserialization that defaults older checkpoints to at least the completed
  trial count.
- Added focused Rust integration coverage for pending `ask()` batches crossing
  the exploration boundary and for full-checkpoint save/load after pending
  asks.
- Added Python coverage for `Study.save()` + `Study.load()` continuing the
  AutoStrategy issued-trial budget after pending asks.

## Acceptance Criteria

- Issuing N concurrent `ask()` calls crosses the exploration boundary exactly
  once based on issued trials.
- Checkpoint save/load preserves the issued counter.
- Existing sequential behavior remains unchanged.

## Suggested Tests

- Configure `exploration_budget = 2`, call `ask()` 4 times before any `tell()`,
  assert first 2 come from Sobol path and later suggestions use GMM path.
- Save after pending asks, load, continue asking, assert budget continuity.

## Verification

- `cargo test -p hola --test integration auto_strategy --all-features`
- `cargo test -p hola --test integration checkpoint --all-features`
- `cd hola-py && uv run maturin develop`
- `cd hola-py && uv run pytest -q tests/test_study_advanced.py -k 'gmm_counts_pending_asks or gmm_save_load_preserves_pending_ask_accounting'`
- `cargo test --workspace --all-features`
- `cd hola-py && uv run pytest -q`

All passed after the fix.
