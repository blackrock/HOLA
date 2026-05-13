# Ranking And Refit Paths Scale Poorly Under Locks

## Status

Resolved in branch `audit/known-issues` with the issue 09 fix.

## Tags

- severity: medium
- type: performance
- area: leaderboard
- area: concurrency
- area: dashboard
- area: multi-objective
- regression-risk: medium

## Summary

Several hot paths rebuild ranked views by cloning and sorting the entire
leaderboard while engine locks are held or immediately after mutating state.
Multi-objective ranking is documented as `O(M * N^2)`. This is acceptable for
small studies, but will become a bottleneck for large dashboards, frequent
SSE-driven refreshes, and distributed ask/tell workloads.

## Evidence

- `tell()` computes a completed view immediately after pushing a trial.
  - `hola/src/hola_engine.rs:1157`
- `get_completed()` builds full ranked lists and searches them.
  - `hola/src/hola_engine.rs:714`
- Multi-objective non-dominated sort is documented as `O(M * N^2)`.
  - `opt_engine/src/leaderboard.rs:653`
- Dashboard refetches all trials on each SSE trial completion.
  - `dashboard/app.js:77`

## Impact

- `tell()` latency grows with leaderboard size.
- Dashboard-connected runs can repeatedly trigger full ranking and JSON
  serialization.
- Multi-objective studies can become slow at hundreds or thousands of trials.
- Long read/write lock holds can block concurrent asks/tells.

## Proposed Fix

Separate mutation, ranking, and presentation concerns.

Recommended design:

- In `tell()`, return the just completed trial with lightweight rank info or
  compute rank outside the write lock using a cloned snapshot.
- Add cached ranking snapshots invalidated on leaderboard mutation.
- Add paginated or incremental `/api/trials` responses.
- Add `/api/trial/{id}` so remote Python `tell()` does not fetch every trial.
- Consider specialized Pareto/ranking data structures for large
  multi-objective studies.

## Resolution

- Changed `HolaEngine::tell()` to snapshot the leaderboard and release the
  write lock before constructing the ranked `CompletedTrial`.
- Added scalar single-trial completion lookup that computes exact rank with a
  linear scan instead of rebuilding the full ranked list.
- Added `HolaEngine::completed_trial()` and `GET /api/trial/{trial_id}` for
  single completed-trial retrieval.
- Included the completed trial in `POST /api/tell` responses and SSE
  `TrialCompleted` events.
- Updated remote Python `Study.tell()` to consume the returned completed trial,
  falling back to `/api/trial/{id}` and only then to the legacy full-trials
  fetch for older servers.
- Updated the dashboard SSE path to upsert a single completed trial instead of
  refetching all trials after each completion.
- Optimized scalar `top_k`/`bottom_k` and vector `top_k_scalarized` to select
  only the requested prefix before sorting it, avoiding full sorts for refit
  and small top-k requests.
- Added ignored Rust scalability probes for representative scalar and vector
  leaderboard ranking sizes.

## Acceptance Criteria

- `tell()` does not rebuild the entire ranked list while holding the write lock.
- Dashboard can update from a single completed-trial event or paginated fetch.
- Benchmarks exist for scalar and vector leaderboard ranking at representative
  sizes, for example 1k, 10k, and 50k scalar trials.

## Suggested Tests

- Add criterion or lightweight benchmark tests for `top_k`, `trials`, and
  `pareto_front`.
- Add a concurrency test with many asks/tells and dashboard reads to catch lock
  contention regressions.

## Verification

- `cargo test -p opt_engine leaderboard::tests::test_top_k --all-features`
- `cargo test -p opt_engine --test integration leaderboard --all-features`
- `cargo test -p hola --test integration server --features server`
- `cd hola-py && uv run maturin develop`
- `cd hola-py && uv run pytest -q tests/test_server.py -k 'ask_tell_best_flow or study_connect_ask_tell_best'`
- `cargo test --workspace --all-features`
- `cd hola-py && uv run pytest -q`

All passed after the fix.
