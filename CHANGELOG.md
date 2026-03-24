# Changelog

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
