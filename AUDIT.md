# Codebase Audit Tracker

This document is the durable reference for the codebase audit performed and
remediated on 2026-07-10. Finding IDs are stable and should be referenced in
commits, pull requests, changelog entries, and follow-up tests.

The findings were originally recorded against the pre-rename
`epicycloids/robopt` fork. Remediation was rebased onto the canonical
`blackrock/HOLA` repository before implementation. Historical evidence links
below therefore preserve the reviewed names and paths; the closure matrix maps
each finding to the canonical implementation and verification evidence.

## Audit Metadata

| Field | Value |
| --- | --- |
| Audit date | 2026-07-10 |
| Original audit repository/commit | `epicycloids/robopt` @ `3ca4169` |
| Canonical remediation repository | `blackrock/HOLA` |
| Canonical baseline | `cbd17256cf950fccc4d7828ab6475002b4c22747` (`hola-upstream/main`) |
| Remediation branch | `wip` |
| Remediation form | Local worktree; commit/PR pending |
| Local Rust toolchain | `rustc 1.92.0`, `cargo 1.92.0` |
| MSRV verified | Rust 1.87.0 |
| Local Python | CPython 3.14 |
| Closure date | 2026-07-10 |

| Severity | Original findings | Open after remediation |
| --- | ---: | ---: |
| Critical | 3 | 0 |
| High | 24 | 0 |
| Medium | 29 | 0 |
| Low | 1 | 0 |
| **Total** | **57** | **0** |

The audit covered security boundaries, correctness, concurrency, numerical
stability, persistence and recovery, distributed-job semantics, performance,
API ergonomics, Python and CLI behavior, dashboard behavior, CI and GitHub
Actions, dependencies, portability, packaging, documentation, observability,
accessibility, and test quality.

## Tracking Conventions

- `[ ]` means open.
- `[x]` means fixed and verified locally, with any unavailable platform/browser
  verification encoded as a required hosted or release gate.
- Do not delete closed findings. The closure matrix below is the fixing and
  verification record until the worktree is committed and a PR exists.
- If a fix intentionally changes scope, record the decision under the finding.
- Severity reflects the reviewed deployment model, including a network-reachable
  distributed server. A strictly local-only deployment may lower some security
  severities, but does not remove the underlying defects.

## Remediation Order (Completed)

1. Secure the service boundary and dashboard: `SEC-001`, `SEC-002`.
2. Repair broken data contracts: `CORE-001`, `CORE-002`, `DATA-001`, `DATA-002`.
3. Restore trustworthy automation: `CI-001`, `CI-002`, `CI-003`.
4. Fix concurrency, validation, and job lifecycle: `CORE-003` through
   `CORE-011`, `DIST-001`, `DATA-003`.
5. Address scaling, compatibility, UI correctness, documentation, and remaining
   test gaps.

## Findings

### Security

- [x] **SEC-001 [Critical] - The network server exposes unauthenticated control and arbitrary file replacement.**
  - Evidence: [checkpoint handler](opt_engine/src/server.rs#L201), [permissive CORS](opt_engine/src/server.rs#L254), and [bind on all interfaces](opt_engine/src/server.rs#L271).
  - Impact: Any reachable client, including browser JavaScript, can read study data, issue unlimited trials, change objectives, and replace any service-writable file with checkpoint JSON.
  - Remediation: Default to loopback; add authenticated, role-scoped access; restrict CORS; generate checkpoint names server-side inside a configured directory; add request limits.
  - Done when: Unauthorized reads and mutations are rejected, cross-origin requests are restricted, and path traversal/arbitrary-path tests pass.

- [x] **SEC-002 [High] - Worker and checkpoint data can execute stored script in the dashboard.**
  - Evidence: Untrusted names and values enter `innerHTML` in [Pareto options](dashboard/app.js#L251), [table rendering](dashboard/app.js#L493), and [objective controls](dashboard/app.js#L548).
  - Impact: A malicious worker, checkpoint, or connected server can execute HTML/JavaScript in the dashboard origin.
  - Remediation: Create DOM nodes with `textContent`, use `addEventListener`, remove inline handlers, validate field names, and add a restrictive Content Security Policy.
  - Done when: Automated payload tests for parameter names, metric names, objective names, and string values cannot create executable DOM.

- [x] **SEC-003 [Medium] - GitHub Actions dependencies are mutable and incompletely permissioned.**
  - Evidence: Workflows use mutable major tags in [ci.yml](.github/workflows/ci.yml#L22) and [audit.yml](.github/workflows/audit.yml#L14), and do not declare minimal top-level permissions.
  - Impact: Action behavior can change without a repository change, and token capabilities are determined by repository defaults.
  - Remediation: Upgrade to Node 24-compatible action releases, pin full commit SHAs, and declare least-privilege `permissions` per job.
  - Done when: Every action is SHA-pinned, automated updates are configured, and token permissions are explicit.

### Core Correctness And Numerical Behavior

- [x] **CORE-001 [Critical] - Python and the shipped YAML double-transform log10 bounds.**
  - Evidence: Python converts actual bounds to exponents in [`Log10::new`](robopt-py/src/lib.rs#L43), passes them as `log10` bounds in [parameter extraction](robopt-py/src/lib.rs#L178), and the core applies `log10` again in [continuous normalization](opt_engine/src/spaces/continuous.rs#L64). The shipped [YAML example](robopt-cli/examples/example_study.yaml#L4) also uses exponent bounds.
  - Impact: Log-scaled asks return JSON `null` and Python `None`; two Python tests fail and the shipped CLI example is unusable.
  - Remediation: Keep user-facing bounds as positive actual values and apply the scale exactly once. Validate positivity at construction/config load.
  - Done when: Python, YAML, Rust, REST, and checkpoint round trips all return finite values within the requested actual interval.

- [x] **CORE-002 [High] - Ask IDs and completed-trial IDs diverge when results arrive out of order.**
  - Evidence: Ask allocates from `next_pending_id` in [DynEngine state](opt_engine/src/dyn_engine.rs#L436), while tell appends through the leaderboard's independent counter in [`push_with_raw`](opt_engine/src/leaderboard.rs#L133).
  - Impact: A job asked as ID 1 can appear in history and `best()` as ID 0, breaking worker correlation, events, retries, dashboards, and checkpoints.
  - Remediation: Use one ID authority and persist the issued ID in the completed record. Preserve identity across every API and storage format.
  - Done when: Out-of-order and concurrent tell tests prove the ask ID is unchanged in best, leaderboard, events, exports, and resumed checkpoints.

- [x] **CORE-003 [High] - Objective updates race with tell and can create mixed scalarization.**
  - Evidence: Tell scalarizes before acquiring engine state in [`tell`](opt_engine/src/dyn_engine.rs#L515), while [`update_objectives`](opt_engine/src/dyn_engine.rs#L627) swaps objectives and re-scalarizes in separate critical sections.
  - Impact: A result scored under old objectives can be appended after historical trials were re-scored under new objectives.
  - Remediation: Store objectives with engine state under one synchronization boundary or use objective epochs and re-check before commit.
  - Done when: A deterministic race test cannot produce observations from two objective epochs in one committed snapshot.

- [x] **CORE-004 [High] - Off-lock refitting can overwrite newer strategy state.**
  - Evidence: Generic and dynamic refits clone a strategy, fit off-lock, then replace live state in [generic refit](opt_engine/src/engine.rs#L301) and [dynamic tell refit](opt_engine/src/dyn_engine.rs#L536).
  - Impact: Concurrent asks advance counters and concurrent refits finish out of order; a stale fit can rewind the counter, duplicate suggestions, or discard a newer model.
  - Remediation: Version model state and reject stale fits, or update only fitted parameters while preserving live counters. Serialize concurrent refit commits.
  - Done when: Stress tests with concurrent asks, tells, and delayed refits show no repeated sequence positions or stale-model commits.

- [x] **CORE-005 [High] - Typed Engine ingestion trusts arbitrary candidate JSON.**
  - Evidence: [`Engine::ingest_result`](opt_engine/src/engine.rs#L133) deserializes the candidate but does not call `space.contains` or correlate it with a dispatched job.
  - Impact: Forged, duplicated, stale, or out-of-bounds candidates can poison the leaderboard and strategy.
  - Remediation: Reject candidates outside the configured space and prefer an engine-issued opaque job token over echoing candidate JSON as identity.
  - Done when: Forged, stale, duplicate, and out-of-space ingestion tests fail without mutating strategy or history.

- [x] **CORE-006 [High] - Sobol usage exceeds hard dependency limits without validation.**
  - Evidence: [`SobolStrategy::suggest`](opt_engine/src/strategies/sobol.rs#L113) passes an unbounded index and dimension to `sobol_burley`, which supports only 65,536 samples and 256 dimensions.
  - Impact: The 65,537th suggestion panics in debug and is unspecified in release; dimension 257 panics in all builds.
  - Remediation: Enforce supported limits, segment/reseed safely, or replace the dependency with one designed for the supported study sizes and precision.
  - Done when: Boundary tests cover 65,535/65,536/65,537 samples and 255/256/257 dimensions with documented behavior and no unexpected panic.

- [x] **CORE-007 [High] - The default GMM workflow fails its deterministic quality baseline.**
  - Evidence: The ignored [sphere GMM test](opt_engine/tests/integration/end_to_end.rs#L111) returns a best score of `7.216910833385868` after 500 trials against an expected `< 0.1`. Early refit behavior begins in [GMM refit](opt_engine/src/strategies/gmm.rs#L708).
  - Impact: Early three-component fitting on three elites collapses covariance around sparse initial points, clipping adds boundary mass, and exploration disappears.
  - Remediation: Require enough samples per component, retain a prior/uniform exploration mixture, warm up with Sobol, and revisit censored versus truncated sampling.
  - Done when: A non-ignored deterministic regression passes across debug/release and multiple supported dimensions without weakening the meaningful threshold.

- [x] **CORE-008 [High] - ContinuousSpace accepts invalid bounds and mishandles fixed/tiny ranges.**
  - Evidence: Constructors in [continuous.rs](opt_engine/src/spaces/continuous.rs#L26) accept reversed, non-finite, and non-positive log bounds; fixed ranges compute `0/0` in [`to_unit_cube`](opt_engine/src/spaces/continuous.rs#L64); the absolute `1e-9` containment epsilon is larger than valid tiny ranges.
  - Impact: Invalid configurations panic later or emit NaN/null; fixed dimensions poison GMM refits; tiny spaces accept materially out-of-range values.
  - Remediation: Add fallible validated constructors/custom deserialization, define fixed dimensions as unit value `0.5`, and use exact or scale-relative range checks followed by clamping.
  - Done when: Property tests cover reversed, NaN, infinite, log-invalid, fixed, sub-`1e-9`-scale, and extreme finite ranges.

- [x] **CORE-009 [High] - RefitConfig permits panic-producing and nonsensical values.**
  - Evidence: [`should_refit`](opt_engine/src/traits.rs#L254) divides by `refit_interval`, while constructors accept zero intervals and invalid quantiles.
  - Impact: A zero interval panics after a result may already be committed; negative, zero, NaN, or oversized selection settings produce empty or surprising fits.
  - Remediation: Use `NonZeroUsize`, validate `top_k > 0`, and require finite `0 < quantile <= 1`.
  - Done when: Invalid configurations fail before engine construction and cannot partially commit a tell.

- [x] **CORE-010 [High] - GMM fitting and deserialization are panic-prone despite fallible-looking APIs.**
  - Evidence: [`GmmParams::fit`](opt_engine/src/strategies/gmm.rs#L260) does not validate rectangular, nonempty-dimensional, finite unit-cube samples or fit parameters; Gaussian reconstruction uses `expect` in [deserialization](opt_engine/src/strategies/gmm.rs#L163).
  - Impact: Empty-dimensional and ragged inputs panic; malformed checkpoints can panic instead of returning an I/O/serde error; invalid covariance/variance values bypass invariants.
  - Remediation: Return `Result`, validate all dimensions and finite values, make fields private, and use `#[serde(try_from)]` for validated reconstruction.
  - Done when: Malformed samples, covariance matrices, weights, and checkpoint payloads produce structured errors and never unwind.

- [x] **CORE-011 [High] - Dynamic configuration silently changes typos into valid but wrong studies or panics despite returning Result.**
  - Evidence: Unknown scales and strategies fall back to linear/Sobol in [`from_config`](opt_engine/src/dyn_engine.rs#L445), while invalid discrete/category constructors can panic. Objective directions are strings and unknown values fall through scalarization.
  - Impact: Misspelled configuration can optimize the wrong domain/objective without a startup error.
  - Remediation: Replace strings with serde enums and explicit aliases, use `deny_unknown_fields`, validate the full study with field-path errors, and eliminate constructor panics on config input.
  - Done when: Unknown fields, scales, strategies, objective types, incomplete target/limit pairs, duplicate names, and invalid bounds all fail startup clearly.

- [x] **CORE-012 [Medium] - Typed and dynamic TLP semantics disagree.**
  - Evidence: Typed TLP treats exactly-at-limit as finite priority in [json_tlp.rs](opt_engine/src/transformers/json_tlp.rs#L58), while dynamic scalarization treats the same value as infinite in [dyn_engine.rs](opt_engine/src/dyn_engine.rs#L792). Direction checks are only `debug_assert!`; zero weight times infinity can produce NaN in [grouped TLP](opt_engine/src/transformers/json_tlp_grouped.rs#L218).
  - Impact: The same objective behaves differently across typed Rust, Python, REST, and debug/release builds.
  - Remediation: Centralize a validated objective enum and scalarizer, define exact boundary behavior once, and handle zero-weight infeasible terms explicitly.
  - Done when: One shared conformance test table passes through every frontend and build profile.

- [x] **CORE-013 [Medium] - Discrete and categorical mappings do not preserve their advertised invariants.**
  - Evidence: [`DiscreteSpace::cardinality`](opt_engine/src/spaces/discrete.rs#L25) can overflow `i64`, large ranges exceed `f64` exactness, and [categorical mapping](opt_engine/src/spaces/categorical.rs#L59) accepts duplicate labels. Public fields and derived deserialization bypass constructor checks.
  - Impact: Extreme ranges panic or map incorrectly; duplicate categories are biased and not bijective; malformed serialized spaces can bypass invariants.
  - Remediation: Use checked wider arithmetic, reject unsupported cardinalities and duplicates, keep fields private, and validate deserialization.
  - Done when: Boundary/property tests prove round trips or return an explicit unsupported-range error.

- [x] **CORE-014 [Medium] - Leaderboard total ordering and multi-objective schema behavior are undefined for malformed/non-finite data.**
  - Evidence: `_all` ranking uses `partial_cmp(...).unwrap_or(Equal)` in [leaderboard.rs](opt_engine/src/leaderboard.rs#L259); crowding takes objective keys only from the first trial in [`crowding_distance`](opt_engine/src/leaderboard.rs#L748); empty maps are vacuously feasible.
  - Impact: NaN or inconsistent objective maps can rank before finite trials, ignore later objectives, or appear Pareto-feasible.
  - Remediation: Enforce a common objective schema and explicit total ordering for finite, infeasible, and invalid observations.
  - Done when: Heterogeneous maps, empty maps, NaN, and both infinities have documented deterministic rankings or are rejected.

- [x] **CORE-015 [Medium] - Configuring auto-refit does not make normal ingestion auto-refit.**
  - Evidence: The builder advertises automatic refitting in [engine.rs](opt_engine/src/engine.rs#L452), but callers must discover and use separate [`ingest_result_with_refit`](opt_engine/src/engine.rs#L345). A refit failure is returned after ingestion has committed.
  - Impact: Users can configure refit and silently never refit; retrying a reported failure can duplicate an already committed result.
  - Remediation: Consolidate ingestion semantics or make refitting explicit in the engine type. Return a result that distinguishes committed ingestion from failed post-processing.
  - Done when: The primary documented ingestion path honors configuration and failure tests prove retry semantics.

- [x] **CORE-016 [Medium] - Dynamic studies cannot configure or recover deterministic seeds consistently.**
  - Evidence: [`StrategyConfig`](opt_engine/src/dyn_engine.rs#L376) has no seed; random and GMM auto-seed while Sobol is hard-coded to 42 in [`from_config`](opt_engine/src/dyn_engine.rs#L465).
  - Impact: Experiments are inconsistently reproducible and the resolved seed is not visible in metadata.
  - Remediation: Accept an optional seed, generate one once when absent, expose it, and persist it with study configuration.
  - Done when: Every strategy can be reproduced from exported study metadata and checkpoint state.

- [x] **CORE-017 [Low] - DynSpace builder and standardized-space length contracts are surprising.**
  - Evidence: Extending a cloned `DynSpace` panics because builders require an Arc refcount of one in [dyn_engine.rs](opt_engine/src/dyn_engine.rs#L184). The trait promises wrong-length rejection in [traits.rs](opt_engine/src/traits.rs#L65), but dynamic/branching spaces accept trailing dimensions.
  - Impact: Valid-looking builder composition can panic, and custom strategies cannot rely on one dimensionality contract.
  - Remediation: Use a separate mutable builder or copy-on-write storage and standardize exact versus prefix length behavior.
  - Done when: Clone-then-extend works or is a compile-time/API error, and all spaces pass one shared dimensionality contract suite.

### Persistence And Data Integrity

- [x] **DATA-001 [Critical] - JSON checkpoints cannot round-trip intentional non-finite observations.**
  - Evidence: [checkpoint JSON writing](opt_engine/src/persistence.rs#L115) serializes `Infinity`, `-Infinity`, and `NaN` as `null`; [loading](opt_engine/src/persistence.rs#L121) cannot deserialize `null` into `f64`.
  - Impact: A checkpoint containing a valid infeasible TLP result reports save success but is permanently unloadable.
  - Remediation: Persist a tagged score/status representation or use explicit lossless float serde. Never silently convert to null.
  - Done when: Scalar and multi-objective file/string round trips preserve finite and every supported non-finite/status value.

- [x] **DATA-002 [High] - Full checkpoints omit state required for safe resumption.**
  - Evidence: [`Checkpoint`](opt_engine/src/persistence.rs#L86) stores only leaderboard, strategy, and metadata; dynamic state also includes pending jobs, next ID, objectives, refit configuration, and the space schema in [dyn_engine.rs](opt_engine/src/dyn_engine.rs#L428).
  - Impact: Restarts can reuse IDs, reject late workers, mix historical/new objective definitions, or load an incompatible GMM and panic on ask.
  - Remediation: Define a versioned `DynEngineCheckpoint` containing all required state and a space/objective/config fingerprint, or explicitly quiesce/invalidate pending work with an epoch.
  - Done when: Resume tests cover pending work, updated objectives, every strategy, ID continuity, and incompatible-space rejection.

- [x] **DATA-003 [High] - Atomic checkpoint writes are unsafe under concurrency and incomplete across platforms.**
  - Evidence: [`atomic_write_json`](opt_engine/src/persistence.rs#L211) always uses `path.with_extension("tmp")`.
  - Impact: Concurrent writers share and truncate one temporary file; destinations ending in `.tmp` are written in place; Windows replacement may fail when the destination exists; the parent directory is not fsynced.
  - Remediation: Use a unique `create_new` temporary file in the destination directory, coordinate same-path writers, implement cross-platform atomic replacement, clean failures, and fsync the directory.
  - Done when: Concurrent overwrite, crash/failure cleanup, repeated Windows save, and durability-oriented tests pass.

- [x] **DATA-004 [Medium] - Checkpoint version and internal invariants are written but not validated.**
  - Evidence: `format_version` is stored in [metadata](opt_engine/src/persistence.rs#L25) but ignored on load; leaderboard deserialization does not validate `n_trials`, unique IDs, or `next_id > max(trial_id)` in [leaderboard.rs](opt_engine/src/leaderboard.rs#L90).
  - Impact: Malformed or future-format checkpoints can create duplicate IDs or partially valid state.
  - Remediation: Validate and migrate a dedicated DTO before mutating live engine state. Reject unsupported versions and invariant violations.
  - Done when: Corrupt, future-version, duplicate-ID, stale-next-ID, and mismatched-count fixtures all fail atomically.

- [x] **DATA-005 [Medium] - Auto-checkpoint configuration and snapshot reporting are unreliable.**
  - Evidence: The configured directory is not created; zero interval panics in [`should_checkpoint`](opt_engine/src/persistence.rs#L268); rotation sorts filenames lexically in [dyn_engine.rs](opt_engine/src/dyn_engine.rs#L739); the HTTP handler recounts trials after writing in [server.rs](opt_engine/src/server.rs#L201).
  - Impact: Documented defaults can fail, million-plus filenames rotate incorrectly, and a checkpoint named/reported for N trials can contain a later concurrent snapshot.
  - Remediation: Validate at startup, create the directory, rotate by parsed sequence/metadata, and return metadata captured from the actual snapshot.
  - Done when: Directory creation, zero/max-zero policy, large sequence numbers, concurrent tells, and rotation tests pass.

- [x] **DATA-006 [High] - Async checkpoint APIs perform blocking serialization and filesystem work.**
  - Evidence: [generic saves](opt_engine/src/engine.rs#L206) and [dynamic saves](opt_engine/src/dyn_engine.rs#L684) synchronously pretty-print, flush, fsync, rename, and rotate from async call paths.
  - Impact: Large checkpoints stall Tokio workers and unrelated HTTP requests; generic saves can retain state access during the write.
  - Remediation: Snapshot briefly, then serialize and persist via `spawn_blocking` or a dedicated bounded persistence service.
  - Done when: A large-checkpoint load test shows bounded request latency and no engine lock held during disk I/O.

- [x] **DATA-007 [Medium] - Generic checkpoint restore can partially succeed contrary to its contract.**
  - Evidence: [`Engine::load_checkpoint`](opt_engine/src/engine.rs#L274) silently discards checkpoint leaderboard history if the target engine was built without a leaderboard while still restoring strategy state.
  - Impact: Callers receive success after only part of the advertised full state was restored.
  - Remediation: Reject incompatible engine configuration or make partial restore a separate explicitly named API.
  - Done when: Restore either applies all declared state or returns an error without mutation.

### Distributed Execution And CLI

- [x] **DIST-001 [High] - The job lifecycle is unbounded, non-idempotent, and not recoverable.**
  - Evidence: Every [`ask`](opt_engine/src/dyn_engine.rs#L504) inserts a permanent pending entry; there are no leases, timeouts, failure/cancel states, quotas, heartbeats, retry keys, or pending checkpoint state.
  - Impact: Worker crashes leak pending work and memory; network retries can create orphan jobs or turn a successful tell into an `unknown trial` error.
  - Remediation: Model explicit job states and leases, bound pending work, add idempotency keys, define duplicate-result semantics, and persist/invalidate leases on restart.
  - Done when: Worker crash, timeout, duplicate ask/tell, uncertain response, cancellation, and restart scenarios are deterministic and bounded.

- [x] **DIST-002 [High] - The CLI worker can corrupt study history and lose results.**
  - Evidence: [worker mode](robopt-cli/src/main.rs#L55) ignores ask/tell HTTP status, defaults a missing ID to zero, ignores subprocess exit status/stderr, converts malformed stdout to `{"error":"parse_failed"}`, and never retries a lost tell.
  - Impact: Failed jobs are recorded as completed infeasible trials, server errors print as success, and transient network failure permanently loses expensive work.
  - Remediation: Validate every response schema/status, require successful subprocess exit, separate logs from the metrics channel, add request/command timeouts, and persistently retry tell with idempotency.
  - Done when: Tests cover HTTP errors, missing fields, nonzero exit, invalid/large output, command timeout, tell retry, and process termination.

- [x] **DIST-003 [Medium] - Server events and response metadata misstate what happened.**
  - Evidence: `TrialCompleted.score` is the global best rather than the completed trial's score in [server.rs](opt_engine/src/server.rs#L98); `RefitOccurred` is defined but never emitted in [EngineEvent](opt_engine/src/server.rs#L43); checkpoint counts can come from a later snapshot.
  - Impact: Consumers cannot trust event payloads for incremental state and telemetry.
  - Remediation: Emit data from the committed operation result, include event/study epochs, and either emit or remove advertised event variants.
  - Done when: Contract tests compare event and response payloads with the exact committed trial/snapshot under concurrency.

### Performance And Resource Use

- [x] **PERF-001 [High] - Live updates become superlinear with study size.**
  - Evidence: Every tell computes best by sorting all feasible trials through [`top_k(1)`](opt_engine/src/leaderboard.rs#L237); every SSE event refetches the full leaderboard in [app.js](dashboard/app.js#L64); each render recomputes an O(n^2) Pareto front in [`computeParetoFront`](dashboard/app.js#L373).
  - Impact: Network traffic, allocation, sorting, JSON cloning, and browser work grow rapidly even when no consumer needs a full refresh.
  - Remediation: Maintain best incrementally, emit completed-trial deltas, paginate history, debounce batches, virtualize rows, and use an O(n log n) 2-D front algorithm.
  - Done when: Benchmarks at 1k, 10k, and 100k trials establish bounded per-tell and incremental-render costs.

- [x] **PERF-002 [Medium] - Repeated GMM refitting has roughly quadratic cumulative cost.**
  - Evidence: Each refit fully sorts/clones a growing leaderboard in [top_k](opt_engine/src/leaderboard.rs#L237), then runs up to 100 EM iterations over a growing elite fraction in [GMM fit](opt_engine/src/strategies/gmm.rs#L260).
  - Impact: Long studies spend increasing time selecting and repeatedly reprocessing old trials; dynamic selection deep-clones JSON that fitting discards.
  - Remediation: Use partial selection or an elite heap, cap/weight the training window, move selection into the blocking task, or implement incremental fitting.
  - Done when: A refit benchmark demonstrates the chosen asymptotic and memory bounds as history grows.

- [x] **PERF-003 [Medium] - GMM hot paths allocate despite zero-allocation claims.**
  - Evidence: [`log_pdf`](opt_engine/src/strategies/gmm.rs#L133) allocates a difference and triangular-solve result per sample/component; [`sample_unclamped`](opt_engine/src/strategies/gmm.rs#L233) rebuilds `WeightedIndex` for every suggestion.
  - Impact: Allocation overhead grows with samples, components, dimensions, and suggestions.
  - Remediation: Reuse scratch buffers/in-place solves and cache the component distribution whenever weights change.
  - Done when: Allocation-count and throughput benchmarks cover fitting and suggestion hot paths.

- [x] **PERF-004 [High] - Every Python Study and RemoteStudy creates a full Tokio runtime.**
  - Evidence: Runtime creation occurs in [Study](robopt-py/src/lib.rs#L297) and [RemoteStudy](robopt-py/src/lib.rs#L525). Locally, each object added 16 threads; three of each produced 97 total process threads.
  - Impact: Multiple studies oversubscribe CPUs and can exhaust process/thread limits before useful work begins.
  - Remediation: Share a process-wide runtime or use a deliberately sized/current-thread runtime with clear lifecycle ownership.
  - Done when: Creating many studies keeps thread count bounded and teardown tests show no leaked runtime resources.

- [x] **PERF-005 [Medium] - Dashboard rendering and fetching have additional large-data failure modes.**
  - Evidence: Each update destroys/recreates uPlot and every table row in [renderAll](dashboard/app.js#L157); multiple event fetches can complete out of order; large `Math.min(...values)` spreads occur in [chart range calculation](dashboard/app.js#L305).
  - Impact: The UI can regress to an older response, freeze on DOM volume, or throw argument-limit errors on large histories.
  - Remediation: Apply monotonic request generations/abort controllers, update charts incrementally, virtualize/paginate, and compute extrema iteratively.
  - Done when: Browser tests remain responsive and monotonic under burst events and six-figure trial counts.

### Python API

- [x] **PY-001 [High] - RemoteStudy blocks the GIL, has no timeout, and masks HTTP errors.**
  - Evidence: [`RemoteStudy::ask`](robopt-py/src/lib.rs#L539) and [`best`](robopt-py/src/lib.rs#L599) call `Runtime::block_on` while holding Python access, parse bodies without `error_for_status`, and use a client without a request timeout.
  - Impact: A slow server can freeze unrelated Python threads indefinitely; a 500 JSON response becomes `missing id` or `None` rather than a server error.
  - Remediation: Release the GIL around blocking work, configure connect/request timeouts, check status before parsing, and preserve structured server errors.
  - Done when: Slow/error mock-server tests prove other Python threads progress and callers receive typed, status-aware exceptions.

- [x] **PY-002 [Medium] - Study.run parallel execution is not exception-safe or adaptively scheduled.**
  - Evidence: [parallel run](robopt-py/src/lib.rs#L386) manually constructs a `ThreadPoolExecutor`, waits in submission order, processes fixed batches, and calls shutdown only on the success path. `n_workers=0` uses raw CPU count.
  - Impact: Objective exceptions can leave threads and pending trials behind; a slow first future blocks completed results; high-core machines create excessive workers.
  - Remediation: Use context-manager/finally cleanup, completion-order collection, bounded defaults, and explicit failed/cancelled trial handling.
  - Done when: Exception, cancellation, slow-first-task, and high-core tests leave no executor threads or stranded pending jobs.

- [x] **PY-003 [Medium] - Python-facing object shapes and errors are inconsistent and effectively untyped.**
  - Evidence: `ask()` returns `Trial(id, params)` while [`best`](robopt-py/src/lib.rs#L324) returns a dict using `trial_id` and `candidate`; runtime, HTTP, schema, and user errors generally become `ValueError`; [ty configuration](robopt-py/pyproject.toml#L34) ignores the native module import and no stubs are shipped.
  - Impact: Users must special-case equivalent records, cannot catch meaningful error classes, and receive little editor/type-checker help.
  - Remediation: Define one immutable trial/result model, a small exception hierarchy, docstrings/signatures, and generated or maintained `.pyi` stubs.
  - Done when: Public API typing tests and documentation examples pass against the built wheel.

### Dashboard And Frontend

- [x] **WEB-001 [High] - Categorical studies cannot render parallel coordinates or sort correctly.**
  - Evidence: The server drops `ParamInfo.choices` in [`handle_space`](opt_engine/src/server.rs#L181), while [parallel coordinates](dashboard/app.js#L413) subtract numeric bounds from string candidates and [table sort](dashboard/app.js#L507) subtracts string values.
  - Impact: Categorical axes produce NaN and categorical column sorting does nothing useful.
  - Remediation: Include choices in the API, map values to stable indices with visible labels, and use type-aware comparators.
  - Done when: Mixed continuous/discrete/categorical visual and sort tests pass.

- [x] **WEB-002 [Medium] - Dashboard Pareto results are wrong for maximize and constrained objectives.**
  - Evidence: [Pareto rendering](dashboard/app.js#L270) always minimizes both selected raw metrics and ignores objective direction, target, limit, and feasibility.
  - Impact: The highlighted front can be the opposite of the study's actual trade-off set.
  - Remediation: Apply objective metadata to orientation and feasibility, or explicitly restrict the UI to two minimization metrics and label that limitation.
  - Done when: Min/min, min/max, max/max, constrained, tied, and infeasible reference datasets produce expected fronts.

- [x] **WEB-003 [Medium] - SSE lag/reconnect and live/offline transitions can leave stale or overwritten state.**
  - Evidence: Broadcast errors are discarded and SSE has no ID/replay/keepalive in [handle_events](opt_engine/src/server.rs#L231); [loading a file](dashboard/app.js#L78) does not close the existing EventSource.
  - Impact: Dropped events are silent, proxy disconnects lack reliable recovery, and a later live event can overwrite an offline file view.
  - Remediation: Close/abort on mode changes, use generation IDs, resync after lag/open, add keepalive and event IDs, and support replay or periodic reconciliation.
  - Done when: Lag, reconnect, duplicate event, out-of-order fetch, and live-to-offline tests converge to the correct state.

- [x] **WEB-004 [Medium] - Dashboard export is not a valid engine checkpoint.**
  - Evidence: [`exportData`](dashboard/app.js#L613) emits `next_trial_id` and omits required `metadata.created_at`; Rust expects `next_id` in [Leaderboard](opt_engine/src/leaderboard.rs#L90) and all metadata fields in [CheckpointMetadata](opt_engine/src/persistence.rs#L25).
  - Impact: A prominently exported JSON file cannot be loaded by the engine as a checkpoint.
  - Remediation: Serialize the exact versioned DTO through shared schema fixtures or label the output explicitly as dashboard-only data.
  - Done when: A dashboard-export fixture loads in Rust and round-trips without field loss.

- [x] **WEB-005 [Medium] - Offline availability, accessibility, and narrow-screen behavior are incomplete.**
  - Evidence: Dashboard startup depends on unpinned CDN JavaScript/fonts without SRI in [index.html](dashboard/index.html#L9); URL input and canvases lack accessible labeling; sortable headers are mouse-only; [layout](dashboard/styles.css#L40) can clip nonwrapping controls.
  - Impact: The documented offline workflow can fail without internet, supply-chain integrity is weak, keyboard/screen-reader operation is incomplete, and small screens can hide controls.
  - Remediation: Vendor or integrity-pin assets, add a CSP, semantic labels and keyboard controls, canvas alternatives, and responsive overflow behavior.
  - Done when: Offline, CSP/SRI, keyboard, screen-reader smoke, and narrow mobile viewport tests pass.

### CI, GitHub Actions, And Dependencies

- [x] **CI-001 [High] - The Rust CI job cannot pass.**
  - Evidence: [Rust CI](.github/workflows/ci.yml#L14) runs format, clippy, and tests with warnings denied. Locally, formatting failed extensively, exact clippy failed with 23 errors, and `RUSTFLAGS=-D warnings` rejected two deprecated PyO3 calls.
  - Impact: Every matrix platform stops before providing meaningful regression confidence.
  - Remediation: Format the tree, fix or intentionally configure lints, replace deprecated PyO3 calls, and keep warning policy scoped to workspace code.
  - Done when: The exact workflow commands pass on Linux, macOS, and Windows from a clean checkout.

- [x] **CI-002 [High] - The Python CI job has failures at every validation layer.**
  - Evidence: [Python CI](.github/workflows/ci.yml#L40) currently encounters 3 Ruff format failures, 9 Ruff lint errors, 1 ty diagnostic, and 2 pytest failures. Its later [`uv run maturin develop`](.github/workflows/ci.yml#L70) cannot find `maturin` because [pyproject](robopt-py/pyproject.toml#L1) lists it only as an isolated build requirement.
  - Impact: The job cannot reach a green state, and fixing one stage only exposes the next failure.
  - Remediation: Fix all diagnostics and Log10 behavior; either rely on `uv sync` installing the project or add/use maturin explicitly in a reproducible build flow.
  - Done when: The exact CI sequence passes from an empty environment and tests the wheel that would be distributed.

- [x] **CI-003 [High] - Dependency auditing is red and the audit workflow cannot report correctly.**
  - Evidence: Current `cargo audit` found 6 advisories and 2 warnings: two direct PyO3 advisories, four locked `rustls-webpki` advisories, unmaintained `paste`, and an unsoundness warning for `rand`. The only recorded [Security Audit run](https://github.com/epicycloids/robopt/actions/runs/23389034658) failed with `Resource not accessible by integration`; [audit.yml](.github/workflows/audit.yml#L9) lacks `checks: write`, and `rustsec/audit-check@v2` uses Node 20.
  - Impact: Known advisories remain in the lockfile while the scheduled control fails for permission/runtime reasons.
  - Remediation: Upgrade PyO3 and the lockfile, assess warning applicability, replace or update the stale audit action, and grant only the permission the chosen reporting design needs.
  - Done when: `cargo audit` is clean or has documented, expiring exceptions and scheduled/PR audit runs succeed.

- [x] **CI-004 [Medium] - CI is not fully reproducible or protected against tool drift.**
  - Evidence: The project and actions float `stable`/major tags in [ci.yml](.github/workflows/ci.yml#L22); cargo commands omit `--locked`; `uv sync` omits `--locked`/`--frozen`; no job timeouts or concurrency cancellation are configured.
  - Impact: Compiler/lint/action behavior can change independently, lock drift can be accepted, and redundant/hung runs consume capacity.
  - Remediation: Pin the Rust version and action SHAs, use locked installs, declare timeouts, and cancel superseded branch/PR runs.
  - Done when: Re-running the same commit resolves identical tool/dependency versions and stale runs are cancelled.

- [x] **CI-005 [Medium] - CI coverage does not match declared compatibility or release surfaces.**
  - Evidence: Python declares `>=3.8` but tests only Linux 3.12; runtime CLI behavior is not tested on Windows despite using `sh`; dashboard, rustdoc, package contents, wheel portability, MSRV, ignored quality tests, and security behavior are absent.
  - Impact: Compatibility, packaging, documentation, and frontend regressions can merge despite a future green core matrix.
  - Remediation: Add supported Python/OS and MSRV matrices, rustdoc/package/wheel checks, dashboard tests, selected security contract tests, and deterministic quality/benchmark gates.
  - Done when: The matrix and release checks explicitly cover every supported platform/runtime advertised in metadata and documentation.

### Packaging And Compatibility

- [x] **PKG-001 [Medium] - Published artifacts omit required project metadata and license files.**
  - Evidence: Cargo packaging warns that manifests lack description/documentation/homepage/repository and package lists omit both license files. The built Python wheel contains no license/readme/summary; [pyproject metadata](robopt-py/pyproject.toml#L5) only supplies name, version, Python floor, and classifiers.
  - Impact: Registry pages and installed artifacts lack provenance, usage context, and bundled license text.
  - Remediation: Add complete workspace/package and PEP 621/639 metadata, include README and dual-license files in every crate/sdist/wheel, and test artifact contents.
  - Done when: Cargo packages and wheel/sdist metadata contain descriptions, links, README, SPDX-compatible license metadata, and both license texts.

- [x] **PKG-002 [Medium] - Declared portability is not enforced and several runtime paths are platform-specific.**
  - Evidence: No Rust `rust-version` is declared although `nalgebra 0.34.1` requires Rust 1.87; [CLI worker](robopt-cli/src/main.rs#L73) hardcodes `sh`; checkpoint replacement differs on Windows; there is no manylinux/macOS/Windows wheel release matrix.
  - Impact: Users can select an edition-compatible but dependency-incompatible compiler, Windows worker mode fails at runtime, and locally built wheels are platform-specific rather than broadly distributable.
  - Remediation: Declare/test MSRV, execute commands without a hardcoded Unix shell or provide platform adapters, fix replacement semantics, and build audited wheels for supported targets.
  - Done when: MSRV, Linux, macOS, and Windows runtime/package smoke tests pass.

- [x] **PKG-003 [Medium] - Dependency and feature choices inflate builds and reduce portability.**
  - Evidence: Core and bindings use `tokio` with `full` features in [Cargo manifests](opt_engine/Cargo.toml#L13), `reqwest` default TLS pulls a broad network stack, and `serde_yaml 0.9` is deprecated. Partial server dependency features are publicly exposed even though only the aggregate server feature is useful.
  - Impact: Compile time, binary/wheel size, thread/runtime surface, and target constraints are larger than necessary.
  - Remediation: Select minimal Tokio/Reqwest features, replace deprecated YAML support, simplify feature flags, and measure artifact size/build time.
  - Done when: Feature-minimal and all-feature builds pass, dependencies are maintained, and size/build benchmarks document the reduction.

### Documentation And Ergonomics

- [x] **DOC-001 [Medium] - README and contribution instructions contain broken commands and paths.**
  - Evidence: [README CLI examples](README.md#L46) use binary `robopt-cli`, `--config`, and `--command`, while the actual binary is `robopt`, config is positional, and the option is `--exec`. Root-level Python commands lack the `robopt-py` working directory; `docs/` and `LICENSE` links do not exist. [CONTRIBUTING](CONTRIBUTING.md#L13) runs `uv sync` from a directory without `pyproject.toml`.
  - Impact: First-run setup, CLI use, Python development, architecture navigation, and license discovery fail as documented.
  - Remediation: Execute every documented command in clean CI fixtures, correct paths/options, and link the real dual-license files and current docs.
  - Done when: A documentation smoke job runs quick-start and contributor commands successfully.

- [x] **DOC-002 [Medium] - Rust documentation is stale and mostly not compile-checked.**
  - Evidence: 10 of 14 doctests are ignored; `cargo doc` with warnings denied fails on private/broken intra-doc links in [dyn_engine.rs](opt_engine/src/dyn_engine.rs#L160) and [traits.rs](opt_engine/src/traits.rs#L93). Workspace docs also collide because CLI and Python library targets are both named `robopt`.
  - Impact: Examples can drift from the API and generated documentation is not a reliable validation surface.
  - Remediation: Convert ignored examples into compiling doctests or `no_run`, repair links, resolve/document target-name collisions, and add rustdoc CI.
  - Done when: Doctests and `RUSTDOCFLAGS=-D warnings cargo doc --workspace --all-features --no-deps` pass.

### Operations And Observability

- [x] **OPS-001 [Medium] - Production failures are difficult to diagnose or recover from.**
  - Evidence: Library/server code relies on `eprintln!` and string errors, has no structured tracing/correlation IDs/metrics, health/readiness endpoint, request timeout layer, or graceful shutdown path. HTTP extractor and engine errors are not represented by stable machine-readable codes.
  - Impact: Distributed worker failures, refits, checkpoint stalls, retries, and degraded service cannot be monitored or handled reliably.
  - Remediation: Integrate structured tracing and metrics, typed error codes, health/readiness, request limits/timeouts, and graceful shutdown with checkpoint/lease policy.
  - Done when: Operational tests and a runbook demonstrate diagnosis and clean shutdown for worker, persistence, and server failures.

### Test Gaps

- [x] **TEST-001 [High] - Core concurrency and numerical boundary contracts lack regression coverage.**
  - Missing coverage: out-of-order IDs, objective/tell races, stale/concurrent refits, Sobol limits, zero intervals, invalid/fixed/tiny spaces, discrete overflow, duplicate categories, malformed GMM/checkpoints, and inconsistent objective schemas.
  - Remediation: Add deterministic unit/property tests and concurrency tests with barriers rather than timing sleeps.
  - Done when: Every `CORE-*` concurrency/numerical fix has a focused failing-before/passing-after test.

- [x] **TEST-002 [Medium] - Persistence and distributed recovery paths lack fault-oriented tests.**
  - Missing coverage: concurrent/failed writes, Windows overwrite, parent-directory durability, non-finite round trips, format migration, post-resume IDs, pending leases, idempotency, retries, SSE lag/replay, auth/CORS, request quotas, and worker process/network failures.
  - Remediation: Add fixture-based corruption/migration tests, filesystem fault injection where practical, actual-network integration tests, and restart scenarios.
  - Done when: Every `DATA-*`, `DIST-*`, and `SEC-001` closure condition is exercised in CI.

- [x] **TEST-003 [Medium] - Dashboard, packaging, performance, and accessibility have no automated regression harness.**
  - Missing coverage: XSS, categorical charts, maximize Pareto, valid exports, live/offline switching, burst/stale events, large datasets, keyboard/mobile accessibility, offline assets, artifact metadata/licenses, and performance budgets.
  - Remediation: Add browser integration/accessibility tests, schema fixtures shared with Rust, package inspection tests, and repeatable benchmarks with recorded budgets.
  - Done when: Every `WEB-*`, `PKG-*`, and `PERF-*` user-facing behavior has automated coverage or a documented manual gate.

## Closure Matrix

All resolutions below are in the local remediation worktree based on canonical
commit `cbd1725`. “Full suite” refers to the verification snapshot immediately
following this table.

| ID | Resolution | Focused evidence |
| --- | --- | --- |
| SEC-001 | Loopback is the default; non-loopback binds require a nonempty admin token; read/admin roles and pre-dispatch Origin enforcement are enforced. Checkpoint-save requests accept only a description, and the server generates unique names beneath its configured directory. | Auth/read-role/CORS/body tests plus generated-name uniqueness and client-path-field rejection. |
| SEC-002 | Dashboard rendering uses inert DOM text/event listeners under a restrictive CSP; no untrusted `innerHTML` or inline handler path remains. | Dashboard XSS DOM harness passes malicious parameter, metric, objective, and value payloads. |
| SEC-003 | All 45 external action references are full-SHA pinned, jobs declare least privilege, and Dependabot covers Actions. | Workflow parser and SHA-policy check pass for all three workflow files. |
| CORE-001 | Log/log10 APIs retain positive user-domain bounds and transform exactly once across Rust, YAML, Python, REST, and checkpoints. | Rust scale/config tests, Python Log10 tests, examples, and checkpoint round trips pass. |
| CORE-002 | The issued trial ID is the sole identity authority and is pushed unchanged into completion history/events/checkpoints. | Out-of-order, concurrent, resume, and server event tests preserve IDs. |
| CORE-003 | Objectives and leaderboard are committed under one state lock; migration/refit serialization is atomic. Retained retry receipts are rebuilt for the new ranking epoch, while evicted receipts whose new rank is unknowable are invalidated. | Barrier-forced race test and `test_objective_update_rescores_retry_receipts_and_remains_loadable`. |
| CORE-004 | Refit commits reconcile fitted parameters with live counters and serialize objective-changing refits. | `test_auto_reconcile_keeps_fitted_model_and_live_counters` and barrier-forced concurrent-refit test. |
| CORE-005 | Canonical HOLA removed arbitrary candidate ingestion; `tell` accepts only an opaque pending ID and validates stored engine-issued candidates on restore. | Unknown, duplicate, cancelled, forged-state, and checkpoint validation tests. |
| CORE-006 | Sobol validates dimensionality and deterministically falls back after the backend sample limit. | Exact 255/256/257 dimension and 65,535/65,536/65,537 draw tables pass. |
| CORE-007 | Auto/GMM uses Sobol warm-up plus periodic exploration and guarded component fitting. | Non-ignored `test_seeded_gmm_meets_sphere_quality_baseline` passes. |
| CORE-008 | Continuous spaces have private fields, fallible validated constructors/serde, fixed-range semantics, and relative containment checks. | Boundary/property tests cover reversed, non-finite, log-invalid, fixed, tiny, and extreme ranges. |
| CORE-009 | Refit configuration is private/fallible and defensively rejects zero intervals, zero top-k, and invalid quantiles before use. | Constructor/serde and no-partial-commit tests pass. |
| CORE-010 | GMM construction, fitting, parameter replacement, and serde return structured `GmmError` values and validate every invariant. | Malformed/ragged/non-finite/covariance/weight/partial-commit tests pass without unwind. |
| CORE-011 | Dynamic DTOs deny unknown fields and reject unknown strategies/scales/types, incomplete contracts, duplicates, and invalid bounds at startup. | Focused config-validation and unknown-field tests pass. |
| CORE-012 | Canonical HOLA uses one validated scalarization path and explicit zero-priority/infeasible handling; the conflicting legacy transformer stack is absent. | Objective conformance tests and zero-priority infinity regressions pass. |
| CORE-013 | Discrete/categorical fields are private, checked arithmetic rejects inexact ranges, duplicate choices fail, and serde revalidates. | Boundary/property/duplicate/deserialization tests pass. |
| CORE-014 | Leaderboard and product APIs share explicit total ordering; malformed schemas and non-finite values have deterministic policy. | NaN/infinity/schema tests plus `test_scalar_single_trial_rank_matches_full_total_order`. |
| CORE-015 | Canonical ingestion is the single `tell` path. Commit/event publication precedes awaitable maintenance; owned refit/checkpoint tasks survive caller cancellation, failures remain successful tells, persist in retry receipts, surface through client warnings, and increment dedicated counters. | Cancellation, refit-failure, warning-as-error, retry, and tell-outcome regressions pass. |
| CORE-016 | Every strategy accepts/exports a resolved seed and full checkpoints preserve sampling state. | Random/Sobol/GMM determinism and generated-seed reproduction tests pass. |
| CORE-017 | Dynamic space extension is copy-on-write and every standardized mapping enforces exact dimensionality. | Clone/extend and wrong-length combinator/dynamic-space tests pass. |
| DATA-001 | Checkpoint format 2 recursively tags `+∞`, `-∞`, and NaN while preserving finite bits; NaN payload canonicalization is accepted on self-load. Public score JSON uses distinct `"inf"`, `"-inf"`, and `"nan"` sentinels, decoded only in numeric score/metric contexts. | Scalar/vector NaN round trips, bitwise finite regression, Python context tests, and 50 pending-replay runs pass. |
| DATA-002 | Full checkpoints contain config, strategy, objectives, pending/cancelled jobs, ID cursor, ask keys, leases, and completion receipts with compatibility validation. | Pending/vector/id-continuity/incompatible-engine/restart tests pass. |
| DATA-003 | Atomic writes use unique same-directory temp files, cross-platform replacement, cleanup, file sync, and parent-directory sync. | Concurrent-save, serialization-failure, unique-temp, repeated save, and OS matrix gates. |
| DATA-004 | Version/DTO validation rejects future formats, exhausted or stale counts/IDs, forged candidates/observations/receipts, incompatible strategy kind/dimension/seed/cursor, and oversized/conflicting transient state before mutation. | Corruption/invariant, sampler-state, and full format-1 migration tests pass. |
| DATA-005 | Auto-checkpoint config validates interval/retention, creates directories, rotates numerically, and names from monotonic completion count. Manual REST saves use timestamp/sequence names and both report exact written-snapshot metadata. | Bounded cadence/rotation/directory/cancellation-shielding and generated-name tests pass. |
| DATA-006 | Saves capture a brief owned snapshot; bulk serialization, parsing, validation, filesystem work, and legacy GMM reconciliation run through `spawn_blocking` without the engine state lock. Loads use a configuration-checked short final swap. | Async save/load tests and 100k tell/refit performance gates pass. |
| DATA-007 | Canonical full restore validates and atomically swaps complete state. Explicit leaderboard-only fallback invalidates transient jobs, advances allocation to a randomized high-bit epoch, and reconciles sampler/GMM history from monotonic completion count rather than sparse IDs. | Incompatible/full/fallback, fresh-ID, and sampler re-import tests pass without mutation on failure. |
| DIST-001 | Pending work is capped and leased; asks and tells are retry-safe; 4,096 bounded persisted completion receipts survive leaderboard eviction/restart. | Lease/heartbeat, 10k ask-key, eviction/restart receipt, cancellation, and duplicate tell tests pass. |
| DIST-002 | CLI validates status/schema/acknowledged identity and warning arrays before deleting durable outbox records; it uses adaptive heartbeats, bounded HTTP/command I/O, process-tree termination, and retry-safe ask/tell IDs. | 32 Rust worker tests plus 14 real CLI subprocess tests; CI runs them on Linux/macOS/Windows. |
| DIST-003 | Tell returns exact commit metadata; duplicate retries do not emit events; commit/event publication is cancellation-safe and ordered. Objective changes emit `ObjectivesChanged`, causing a cursor-safe full client resync. | Count, cancellation, duplicate-event, objective-resync, replay, lag, and payload tests pass. |
| PERF-001 | Fast scalar/2-D ranking, delta SSE, indexed client state, incremental best/front caches, and bounded drawing replace full refetch/quadratic rendering. Exact 3+-group sorting remains quadratic by design and is documented with a required retention bound for large studies. | Final local probe: 100k scalar tell ~3.8 ms, 2-D tell ~370 ms; dashboard 100k+2k burst test ~1.0 s. |
| PERF-002 | Refit selection uses partial selection over at most 16,384 recent candidates and fits at most 4,096 samples, keyed to monotonic completions. | Gated 1k/10k/100k workset probe; final 100k selection ~8.3 ms. |
| PERF-003 | GMM caches `WeightedIndex`; production EM allocates one solve scratch buffer per fit and reuses it across samples/components, while each strategy owns a reusable suggestion buffer. | Production scratch pointer tests; final gated probe: 100k samples ~111 ms, 2k fit ~352 ms. |
| PERF-004 | Python studies share one process-wide `OnceLock` Tokio runtime and release the GIL around blocking remote work. | Multi-study/runtime/thread and remote concurrency tests pass. |
| PERF-005 | Dashboard caches/indexes derived state, caps convergence/Pareto/parallel/table work, guards generations, computes extrema iteratively, and accepts bounded 64 MiB files. | 100k checkpoint, 2k SSE burst, replay, cap-boundary, and stale-response tests pass. |
| PY-001 | Remote calls detach from the GIL, use bounded connect/request timeouts, validate status/schema, authenticate every endpoint, and raise structured errors. | `test_remote_client_hardening.py` mock-server tests pass. |
| PY-002 | `Study.run` schedules by completion, caps default workers, cleans executors in `finally`, and cancels stranded trials on failures. | Exception/cancellation/slow-first/worker-bound tests pass. |
| PY-003 | Public immutable trial/result shapes, read-only stubs, and `HolaError` subclasses cover configuration, checkpoint, remote, and objective failures. | Ruff, ty, stub tests, strict remote params tests, and 146 clean-wheel tests pass. |
| WEB-001 | Space responses retain choices; categorical axes and table sorting use declared choice order. | Mixed continuous/integer/categorical DOM/canvas/sort test passes. |
| WEB-002 | Server ranks only feasible live vector trials on front zero; dashboard also refuses to highlight non-finite vectors. | Min/max/constrained/tied/all-infeasible Rust and dashboard reference tests pass. |
| WEB-003 | SSE has IDs, replay/reset/lag/keepalive; clients take a cursor before snapshots, upsert replays, generation-guard live/offline transitions, and fully resync on `ObjectivesChanged`. | Snapshot-gap, objective-change, reconnect, stale-resync, history-expiry, lag, and replay tests pass. |
| WEB-004 | Export is explicitly a versioned dashboard analysis export, not mislabeled as an engine checkpoint; imports accept supported engine/dashboard shapes. | Export/import schema tests and documentation label the scope. |
| WEB-005 | Dashboard is dependency-free/offline, CSP-compatible, keyboard/ARIA labeled, responsive, reduced-motion aware, and canvas text alternatives are present. | Static accessibility/XSS tests plus the required real-browser release checklist. |
| CI-001 | The exact format, strict Clippy, warnings-as-errors tests, and Rustdoc commands are green; CI runs Rust on Linux/macOS/Windows. | Local exact commands pass; hosted matrix is mandatory on push/PR. |
| CI-002 | CI builds a wheel for every supported Python minor, installs that exact artifact, then runs isolated tests outside the source tree; Ruff/ty/wheel/sdist contents are also gated. | Local maturin build, 241 Python tests, and 146 tests from a clean installed wheel pass. |
| CI-003 | Rust/Python/npm audit jobs use current pinned tooling and trigger on every relevant lockfile. | `cargo audit`, pinned pip-audit, and `npm audit --audit-level=high` are clean. |
| CI-004 | Rust, uv, Node, pip-audit, actions, and lockfiles are pinned; jobs have timeouts/concurrency cancellation. Python minor selectors intentionally track the latest security patch. | Workflow parse, 45-action SHA policy, and locked local commands pass. |
| CI-005 | CI covers three Rust OSes, Python 3.10–3.14, MSRV/minimal features, real CLI workers, docs, dashboard, audits, artifacts, and gated performance probes. | Workflow/release matrices plus the release verification checklist. |
| PKG-001 | Canonical HOLA's Apache-2.0-only policy is reflected in complete Cargo/PEP metadata with README/license files in every artifact. | Cargo package and wheel/sdist inspection script passes. |
| PKG-002 | MSRV 1.87 is declared/tested; worker shells and checkpoint replacement are platform-adapted; release builds/smokes every advertised native target. | MSRV check, cross-platform CLI CI, and five-architecture wheel matrix. |
| PKG-003 | Tokio/Reqwest features are minimized, deprecated YAML is replaced, minimal builds are gated, and artifact size budgets are enforced. | Minimal/all-feature checks; wheel ≤64 MiB and sdist ≤5 MiB package gates; release timing/size checklist. |
| DOC-001 | README/contributor/CLI paths and flags match canonical HOLA; root links are checked and the README optimization executes. | 40 doc-block/link/command tests (4 intentional pseudo-code skips) and 14 CLI smoke tests pass. |
| DOC-002 | Broken links/target collisions are removed; eight Rust examples compile and Rustdoc warnings are fatal in CI. | Eight doctests and strict workspace Rustdoc pass with zero ignored doctests. |
| OPS-001 | Structured request spans carry request IDs; typed errors, health/readiness, completed/retained gauges, `hola_checkpoint_failures_total`, `hola_refit_failures_total`, bounded mutation detachment/timeouts/leases/drain, and warning propagation are documented. | Metrics/error, forced maintenance failure, cancellation, stuck-shutdown, operations, and release-recovery gates. |
| TEST-001 | Every numerical/concurrency fix has focused unit/property coverage; objective/refit races now start from deterministic barriers. | 465 non-doc Rust tests plus explicit boundary/race/quality probes. |
| TEST-002 | Recovery coverage includes concurrent/failing writes, format-1 migration, retry receipts, leases, auth/CORS, real workers, SSE replay/reset/lag, and bounded shutdown. | Rust/Python/CLI suites and platform/manual fault gates. |
| TEST-003 | Dashboard, packaging, performance, offline/accessibility, and release-platform checks are automated where deterministic and otherwise fail-closed in a recorded release checklist. | Twelve dashboard state tests + XSS, wheel/sdist/rebuilt-wheel inspection, gated 100k probes, and `docs/release-checklist.md`. |

Hosted jobs and the manual browser/platform checklist cannot run until this
local worktree is committed and pushed. The first PR must attach those results;
the workflow and release gates are part of this remediation and fail closed.

## Verification Snapshot

These are the final local remediation results on 2026-07-10. Hosted OS,
architecture, and real-browser results remain required PR/release evidence; they
are gates, not waivers.

| Check | Final local result |
| --- | --- |
| Full Rust workspace | Passed: 465 non-doc tests and 8 doctests. Five performance probes are ignored by default and all passed when run explicitly. |
| Rust format, strict Clippy, warnings, and Rustdoc | Passed with warnings denied across the all-feature workspace. |
| MSRV and minimal features | Passed on Rust 1.87.0; `opt_engine` and `hola` also pass their no-default/minimal-feature checks. |
| Rust quality and performance probes | GMM quality passed; at 100k trials scalar tell was ~3.8 ms, two-objective tell ~370 ms, and bounded refit selection ~8.3 ms; 100k GMM samples took ~111 ms and a 2k fit ~352 ms. |
| Python source suite | Passed: 241 tests, 4 intentional skips; Ruff formatting/lint and ty checks passed. |
| Installed-wheel Python suite | Passed: 146 tests from a clean, isolated wheel installation, including bundled dashboard assets. |
| Native CLI | Passed: 32 Rust tests and 14 real subprocess integration tests. |
| Dashboard | XSS harness and 12 state tests passed; the 100k-checkpoint + 2k-event test completed in ~1.0 s. |
| Dependency audits | `cargo audit`, pinned `pip-audit`, and `npm audit --audit-level=high` reported no known vulnerabilities. |
| Package inspection | Cargo packages, wheel, sdist, and a wheel rebuilt from the sdist passed metadata, license, bundled-dashboard, contents, and size-budget verification. |
| Documentation | Passed: 40 executable/link/static checks with 4 intentional pseudo-code skips; strict MkDocs and strict Rustdoc passed. |
| Workflow policy | All 3 workflows parse; all 45 external action references are pinned to full commit SHAs. |
| Checkpoint determinism | The exact finite-float pending replay regression passed 50 consecutive runs. |

## Commit and Hosted Follow-up

The remediation remains an uncommitted local worktree. After committing and
pushing it to `blackrock/HOLA`, attach the hosted CI matrix and the completed
browser, accessibility, mobile, offline, signal-recovery, and package-install
items from `docs/release-checklist.md` to the first pull request.
