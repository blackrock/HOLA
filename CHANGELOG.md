# Changelog

## Unreleased

## 1.0.1

- Fresh GMM studies now use the held-out-calibrated defaults: twice the raw
  warm-up rule before power-of-two rounding, a 12.5% elite fraction, no ongoing
  post-warm-up Sobol' cadence, one mixture component, and a five-sample elite
  floor. Direct low-level GMM constructors and refit configuration use the same
  component cap. Checkpoints that predate these explicit fields continue to
  resolve to the historical defaults when loaded.
- Multi-group GMM refits now select elites by Pareto rank and crowding
  distance, and grouped objectives apply priority weights only within their
  explicit group.
- GMM exploitation now uses seeded Owen-scrambled Gauss--Sobol' samples.
  Each successfully installed GMM starts at the first point of a new
  epoch-specific scramble, so refitting does not consume or strand suggestion
  points and each fixed model receives a balanced sequence prefix. Checkpoint
  format version 3 records the fitted-model epoch and rejects downgrade into a
  binary that would ignore the new sampling state.
  Configurable sample and candidate limits bound refit work while preserving
  deterministic coverage of long retained histories, and batched completions
  can defer repeated multi-objective ranking until the batch is committed.
- Benchmark campaigns now enforce exact completed-evaluation budgets, paired
  seeds, immutable provenance manifests, failure-visible reporting, fixed
  metric scales, and strict result-coverage validation. The suite also adds
  analytic grouped-TLP and sealed-test mixed-space HPO capability studies.
- Updated the documentation dependency lockfile to resolve CVE-2026-61632 in
  `pymdown-extensions`.

## 1.0.1-rc8

- API tokens now protect read endpoints and SSE by default. The explicit
  `--allow-unauthenticated-reads` flag restores the prior open-read behavior
  for trusted deployments.
- Checkpoints now preserve non-finite observations and pending job identity,
  validate internal invariants, and move serialization/filesystem work off the
  async runtime.
- Checkpoint format 2 records lossless floats, pending ask idempotency keys, and
  renewable job leases while retaining migration support for intact version 1
  checkpoints. Finite JSON floats now also round-trip bit-for-bit.
- Distributed asks and tells are idempotent; workers use bounded request and
  command timeouts, durable result outboxes, and process-tree termination.
  Bounded completion receipts preserve exact tell retries after leaderboard
  eviction and restart without emitting duplicate completion events. Callback
  workers reconcile rejected heartbeats and command failures against the
  server's exact trial lifecycle, so a successful tell remains authoritative
  and cannot be followed by a spurious cancellation.
- Servers expose health, readiness, Prometheus metrics, request IDs, graceful
  shutdown, replayable SSE, configurable leases, and a separate read-only role.
  Cross-origin requests are rejected before dispatch and shutdown drain time is
  bounded even for long-lived connections. Committed tell/objective maintenance
  is cancellation-shielded, warning-producing, and separately counted.
- Live SSE connections now close as soon as shutdown begins, while the bounded
  drain deadline remains a fallback for other stuck requests.
- Space, refit, leaderboard, and GMM state now validate invariants at every
  construction/deserialization boundary. Two-objective ranking uses an
  O(N log N) sweep and GMM hot paths reuse allocations and cached distributions.
- Python studies share one runtime, release the GIL around remote work, schedule
  parallel objectives by completion, export typed error subclasses, and bundle
  the versioned dashboard in editable trees, wheels, and source distributions.
- The dashboard is dependency-free/offline-capable, renders untrusted data as
  inert text, preserves categorical checkpoint metadata, closes snapshot/SSE
  races with replay cursors, keeps controls reachable at narrow viewports,
  keeps focus through keyboard table-sort cycles, disambiguates overlapping
  parameter and metric columns, meets AA text contrast, and uses indexed
  incremental caches for bounded six-figure rendering.
- CI pins toolchains and actions, tests the supported OS/Python/MSRV surfaces,
  audits dependencies on the exact release commit, builds release artifacts
  for supported platforms, emits GitHub provenance for every workflow-uploaded
  wheel, source distribution, and CLI archive, and verifies package metadata,
  licenses, docs, and dashboard contracts.

## 1.0.1-rc7

This release candidate continues the audit and hardening pass for the
Rust/Python HOLA 1.0 stack with high-severity correctness, security, and
release-process fixes.

### Fixed

- Refit no longer drops concurrent sampling-counter or model updates
  under concurrency.
- Degenerate fixed parameters (`min == max`) no longer produce `NaN`.
- `Study.run` now cancels orphaned trials and shuts down its executor
  when the objective function fails.
- WFG benchmark functions evaluate via pymoo, so IGD is measured
  against the matching reference front.
- Corrected documentation examples.

### Security and robustness

- Objective validation now rejects configurations whose declared type
  contradicts the target/limit ordering, including `target == limit`.
  This may reject contradictory configurations that were previously
  accepted.
- Added optional read-endpoint and SSE authentication via
  `--require-read-auth`. This is off by default: reads remain open
  unless explicitly enabled.

### Dependencies

- Upgraded PyO3 to 0.29, clearing RUSTSEC-2026-0176 and
  RUSTSEC-2026-0177.
- Declared MSRV 1.87 and added Dependabot coverage for Cargo and Python
  dependencies.

### Build and release

- Enforced locked Cargo builds in CI and release workflows.
- Hardened release provenance with GitHub attestations and a generated
  PEP 503-compatible simple index with hashed release assets.
- Added pip-audit coverage for the Python package.

## 1.0.1-rc6

This release candidate hardens the Rust/Python HOLA 1.0 stack after
the local audit pass.

### Fixed

- Prevented checkpoint restore from reusing public trial IDs after
  loading leaderboard-only or full checkpoints.
- Unified checkpoint save/load behavior so Rust, Python, REST, CLI,
  and dashboard paths restore completed trials, strategy state, and
  configuration consistently.
- Ensured pending and cancelled in-flight trials are cleared
  deliberately after checkpoint load.
- Preserved correct leaderboard behavior when objective topology
  changes between scalar and vector modes.
- Counted issued AutoStrategy suggestions, including pending asks,
  against the GMM exploration budget.

### Security and robustness

- Required bearer-token auth for write-capable REST endpoints when
  configured, and rejected non-local server binds without auth.
- Removed unsafe dashboard rendering of untrusted HTML-like values.
- Validated dynamic study configuration before runtime.
- Removed the orphaned REST API from `opt_engine`; REST serving now
  lives in `hola`.

### Performance and maintenance

- Reduced leaderboard ranking lock pressure for top-k and Pareto
  queries.
- Minimized Tokio feature usage across crates.
- Updated REST, checkpoint, and strategy documentation to match the
  new behavior.

## 1.0.0

This release is a ground-up rewrite. We replaced the original Python
implementation with a Rust optimization engine, exposed it through Python
bindings (PyO3), and added a REST API, a CLI, and a browser dashboard.
There is no migration path from the old API; users of `hola.tune()` or
`hola_serve` should treat this as a new system.

### Python API

We introduce the `Study` class as the primary interface. A study holds a
parameter space, one or more objectives, and a search strategy. The
ask/tell loop drives optimization.

- `Study(space, objectives, strategy, seed)` creates a local, in-process
  study.
- `Study.connect(url)` returns an HTTP client that exposes the same
  methods against a running server.
- `study.serve(port, background)` hosts a REST server from a local study,
  optionally in a background thread.
- `study.run(func, n_trials, n_workers)` automates the ask/tell loop with
  optional parallel evaluation.
- `study.save(path)` and `Study.load(path)` persist and restore the full
  engine state (leaderboard, strategy, and configuration) as JSON.
- `study.top_k(k)`, `study.trials()`, `study.pareto_front()`, and
  `study.trial_count()` inspect results.
- `study.update_objectives(objectives)` changes objectives mid-run and
  rescalarizes all existing trials.
- `study.cancel(trial_id)` cancels a pending trial.

### Parameter spaces

We support three parameter types, composed via `Space(**kwargs)`.

- `Real(min, max, scale)` defines a continuous parameter. The `scale`
  argument accepts `"linear"` (default), `"log"`, or `"log10"`.
- `Integer(min, max)` defines an integer parameter within an inclusive range.
- `Categorical(choices)` defines a choice from a list of string labels.

### Objectives and scalarization

We provide `Minimize` and `Maximize` objective classes with optional
target-limit-priority (TLP) fields. Each objective accepts `target`,
`limit`, `priority`, and `group`. Objectives sharing the same `group` are
summed into one component of a group-cost vector; distinct groups form axes
for Pareto ranking. Infeasible trials (those exceeding a limit) receive a
score of infinity.

### Search strategies

We ship three strategies.

- **GMM** (default) fits a Gaussian mixture model to the top fraction of
  completed trials and samples from the fitted distribution. Configurable
  via `Gmm(refit_interval, elite_fraction, exploration_budget)`.
- **Sobol** uses Owen-scrambled quasi-random sequences for space-filling
  exploration.
- **Random** draws uniform pseudo-random samples.

### REST API and CLI

We provide a JSON REST API (Axum) with endpoints for ask, tell, cancel,
trials, top_k, pareto_front, objectives, space, and checkpoint management.
Server-sent events at `/api/events` stream trial completions and refit
notifications in real time.

The CLI offers two subcommands. `hola serve` starts a server from a YAML
study configuration. `hola worker` polls the server, executes a shell
command for each trial, and manages the trial lifecycle.

### Dashboard

We include a zero-install browser dashboard (static HTML/CSS/JS) with four
visualizations: a convergence plot, a Pareto scatter with hover tooltips,
parallel coordinates (with categorical axis support), and a sortable trial
table. The dashboard connects via SSE for live updates and can load
checkpoint files for offline analysis.

We provide three objective-editing modes in the dashboard. *Preview*
rescalarizes trials client-side without affecting the server. *Reset*
restores the server's original objectives. *Apply to server* sends the new
objectives to the server and changes future sampling behavior.

### Persistence

We save atomic JSON checkpoints that capture the leaderboard, strategy
state, and study configuration. The checkpoint format is self-contained,
so `Study.load(path)` can reconstruct the full engine without additional
arguments.

### Build and distribution

We publish pre-built Python wheels for Linux (x86_64, aarch64), macOS
(Intel, Apple Silicon), and Windows (x86_64). Pre-built CLI binaries are
available from GitHub releases.

### Breaking changes relative to the Python-only HOLA

| What changed | Old system | New system |
|---|---|---|
| Core runtime | Python | Rust with PyO3 bindings |
| Python API | `hola.tune()` | `Study` class with ask/tell |
| Server | `hola_serve` (Flask-like) | `hola serve` (Axum REST + SSE) |
| HTTP endpoints | `/report_request`, `/get_request`, `/get_candidates` | `/api/ask`, `/api/tell`, and others |
| Configuration | JSON files in a directory | Single YAML file |
| Worker protocol | Custom scripts | `HOLA_PARAMS` environment variable |
| License | Dual MIT / Apache-2.0 | Apache-2.0 only |

### Removed

We removed `hola.tune()`, the `hola_serve` server and its HTTP routes,
leaderboard CSV output, the JSON config directory structure, and the MIT
license.
