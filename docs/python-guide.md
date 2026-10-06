# Python Guide

This guide covers the full Python API for HOLA. For installation
instructions, see [Getting Started](getting-started.md).

## Overview

HOLA's Python API centers on the `Study` class, which can operate
in two modes:

| Mode | Where the engine runs | When to use it |
|------|------------------------|----------------|
| **`Study(...)`** | **In your Python process** (Rust engine loaded inside the interpreter) | Notebooks, single-machine scripts, anything that should not depend on a server |
| **`Study.connect(url)`** | **In a running HOLA server** (returns an HTTP client) | Workers on other machines, language-agnostic workers, sharing one study across many processes |

Both modes expose the same optimization methods (`ask`, `tell`, `top_k`,
`cancel`, and `run`). Local studies also provide checkpoint files and server
hosting; remote studies provide lease renewal through `heartbeat`. Choose the
mode based on where the engine should run.

The Python API exposes these classes:

| Class | Purpose |
|-------|---------|
| `Study` | In-process engine. Pass `Space` and objectives here; also provides `Study.connect(url)` for remote. |
| `Space` | Named parameter space builder |
| `Trial` | A pending trial returned by `ask()`, with `.trial_id` and `.params` |
| `CompletedTrial` | A completed trial with `.trial_id`, `.params`, `.metrics`, `.scores`, `.score_vector`, `.rank`, `.pareto_front`, `.completed_at` |
| `Real` | Real-valued parameter with configurable scale (linear, log, log10) |
| `Integer` | Integer parameter within an inclusive range |
| `Categorical` | Choice from a list of string labels |
| `Minimize` | Minimize an objective field |
| `Maximize` | Maximize an objective field |
| `Gmm` | GMM strategy configuration (refit cadence, elite fraction, exploration, and work limits) |
| `Sobol` | Sobol strategy configuration |
| `Random` | Random strategy configuration |

All classes are imported from the `hola_opt` module.

```python
from hola_opt import (
    Study, Space, Trial, CompletedTrial,
    Real, Integer, Categorical,
    Minimize, Maximize,
    Gmm, Sobol, Random,
)
```

## Exceptions

HOLA exposes a small exception hierarchy so callers can distinguish failures
without parsing message text:

| Exception | Meaning |
|-----------|---------|
| `HolaError` | Base class for errors raised by HOLA |
| `ConfigurationError` | Invalid space, objective, strategy, study, URL, or timeout configuration |
| `CheckpointError` | Checkpoint loading or saving failed |
| `RemoteError` | Remote transport, HTTP status, response schema, or protocol failure |
| `ObjectiveError` | Metrics cannot be converted, or a local tell is rejected (for example, conflicting duplicate metrics) |

All five classes subclass `ValueError`, so code written for earlier releases
that catches `ValueError` remains compatible. Exceptions raised by the user's
objective function itself are propagated unchanged, including their traceback.
Missing or non-numeric objective fields are recorded as infeasible completed
trials rather than raising `ObjectiveError`. IEEE-754 values follow the scoring
rules: favorable infinity can meet a TLP target and score zero, while non-finite
resulting scores make a trial infeasible. Cyclic metrics and
metrics with nesting depth beyond 64 are rejected before completion;
the allocation remains pending so you can retry or cancel it.

```python
from hola_opt import ConfigurationError, RemoteError, Study

try:
    remote = Study.connect("https://hola.example.com", request_timeout=30)
    trial = remote.ask()
except ConfigurationError as error:
    print(f"Invalid client configuration: {error}")
except RemoteError as error:
    print(f"Server request failed: {error}")
```

## Defining Parameter Spaces

A `Space` is built by passing parameter builders as keyword
arguments. The keyword names become the parameter names in trial
dicts.

### Real

A real-valued (floating-point) parameter with a configurable
scale. The `scale` keyword argument accepts `"linear"` (default),
`"log"`, or `"log10"`.

**Linear scale** (default). We sample values uniformly from
$[\min, \max]$.

```python
Space(temperature=Real(0.0, 2.0))
```

**Log scale.** For values that span orders of magnitude, we sample
uniformly in $\ln$ space. Both bounds must be positive.

```python
Space(lr=Real(1e-4, 0.1, scale="log"))
```

**Log10 scale.** Similar to log but uses $\log_{10}$ internally.
Both bounds must be positive.

```python
Space(lr=Real(1e-4, 0.1, scale="log10"))
```

`Real(min, max, scale="linear")`: `min` and `max` are specified
in **actual values** (not exponents), regardless of scale.
Internally, HOLA samples uniformly in the chosen scale's
transformed space.

### Integer

An integer parameter within an inclusive range.

```python
Space(layers=Integer(1, 10))
```

`Integer(min, max)`: values are integers from min to max,
inclusive.

### Categorical

A parameter that chooses from a fixed set of string labels.

```python
Space(optimizer=Categorical(["adam", "sgd", "rmsprop"]))
```

`Categorical(choices)`: `choices` is a list of strings. The
selected label is returned as a string in trial params.

### Mixed Spaces

Combine any parameter types in a single space.

```python
space = Space(
    lr=Real(1e-4, 0.1, scale="log10"),
    layers=Integer(1, 10),
    dropout=Real(0.0, 0.5),
    optimizer=Categorical(["adam", "sgd", "rmsprop", "adamw"]),
)
```

## Defining Objectives

Objectives tell HOLA which fields in your metrics dict to
optimize and in which direction.

### Single Objective

```python
objectives = [Minimize("loss")]
```

Your objective function must return a dict containing the field
name (here `"loss"`).

### Maximize

```python
objectives=[Maximize("accuracy")]
```

Internally, maximization is converted to minimization by negating
the value.

### Multi-Objective

Pass multiple objectives to optimize several metrics
simultaneously.

```python
objectives=[
    Minimize("error"),
    Minimize("latency"),
]
```

Because `group` is omitted here, each field becomes its own priority group and
the leaderboard uses Pareto/NSGA-II ranking over the two group costs. HOLA sums
priority-weighted objective contributions only *within* one shared group. To
request scalar ranking for several fields, give them the same `group` label.

### Target-Limit-Priority (TLP) Objectives

For fine-grained control, use `target`, `limit`, and `priority`.

```python
objectives=[
    Minimize("loss", target=0.0, limit=1.0, priority=1.0),
    Minimize("latency", target=100, limit=500, priority=0.5),
]
```

- **target.** The "good enough" value. Trials at or better than
  target score 0 for this objective.
- **limit.** The worst acceptable boundary. At the limit, an objective scores
  `priority`; crossing beyond it makes the trial infeasible and scores infinity.
- **priority.** The objective's score at the limit and its relative weight
  within a group ($P_i$). The linear segment's slope is
  $P_i / (\text{limit} - \text{target})$; `priority` is not itself a slope.

The TLP formula is
$\varphi_i = P_i \times (\text{value} - \text{target}) / (\text{limit} - \text{target})$.

Between target and limit, the score is interpolated linearly and
scaled by `priority`. See
[Concepts: TLP Scalarization](concepts.md#target-limit-priority-tlp)
for the full explanation.

### Priority Groups

To control Pareto axes explicitly, assign objectives to groups using the
`group` parameter.
Objectives in the same group are summed into a single group cost;
distinct groups form the axes of the Pareto ranking.

```python
objectives=[
    Minimize("error", target=0.05, limit=0.5, priority=1.0, group="quality"),
    Minimize("calibration", target=0.01, limit=0.1, priority=0.5, group="quality"),
    Minimize("latency", target=20, limit=100, priority=1.0, group="cost"),
]
```

Here, `"error"` and `"calibration"` share the `"quality"` group;
their TLP scores are summed into a single quality cost. The
`"latency"` objective forms its own `"cost"` group. The Pareto
front is then computed over the two group axes (quality, cost).

When `group` is omitted, each objective defaults to its own group
(keyed by field name). A study with a single group uses scalar
ranking; multiple groups enable Pareto front via
`study.pareto_front()`.

Pass these objective lists to the `Study` constructor as the
`objectives` parameter, as shown in the next section.

## Creating a Study

```python
study = Study(
    space=Space(x=Real(0.0, 1.0)),
    objectives=[Minimize("loss")],
    strategy="gmm",  # default
    seed=42,           # optional: for reproducible runs
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `space` | `Space` | required | The parameter space to search |
| `objectives` | `list` | required | List of `Minimize` / `Maximize` objectives (at least one) |
| `strategy` | `str` or strategy class | `"gmm"` | Search strategy. Pass a string (`"gmm"`, `"sobol"`, `"random"`) for defaults, or a configuration class (`Gmm(...)`, `Sobol()`, `Random()`) for fine-grained control. |
| `seed` | `int` or `None` | `None` | Random seed for reproducibility. When set, the same seed produces the same candidate sequence. |
| `max_trials` | `int` or `None` | `None` | Cap on lifetime completed plus currently pending trials. `ask()` raises when that total reaches the cap. Cancelling pending work frees a slot; it does not reuse its trial ID. The value also supplies `S` for the automatic GMM warm-up. |
| `max_leaderboard_size` | `int` or `None` | `None` | Maximum completed trials retained for inspection, ranking, and fitting. Must be at least 1. Older entries are evicted; lifetime completion count and trial IDs keep increasing. |

Use `max_leaderboard_size` to bound completed-history memory in a long-running
study. `trial_count()` reports lifetime completions, while `trials()` and
`top_k()` inspect retained history. Evicted observations no longer participate
in rankings or GMM fitting. The GMM work limits below bound each refit, rather
than the stored history itself.

## The Ask/Tell Loop

The core optimization loop has two steps:

1. **Ask:** get the next trial to evaluate.
2. **Tell:** report the result.

```python
for i in range(100):
    trial = study.ask()                    # Trial with .trial_id and .params
    metrics = my_function(trial.params)    # Your evaluation code
    study.tell(trial.trial_id, metrics)    # Report results
```

### `study.ask() -> Trial`

Returns a `Trial` object with:

- `trial.trial_id`: a unique integer identifier (monotonically
  increasing, starting from 0).
- `trial.params`: a dict mapping parameter names to values.

```python
trial = study.ask()
print(trial)           # Trial(trial_id=0, params={'x': 0.4321, 'layers': 5})
print(trial.trial_id)  # 0
print(trial.params)    # {'x': 0.4321, 'layers': 5}
```

### `study.tell(trial_id, metrics) -> CompletedTrial`

Reports the result of a trial. `metrics` must be a dict; provide numeric
values for the fields specified in your objectives to obtain feasible scores.
Returns a `CompletedTrial`. Missing or non-numeric objective fields make the
corresponding score infinite. IEEE-754 metrics follow the objective's scoring
rules; check the resulting scores for finiteness when inspecting feasibility.

```python
completed = study.tell(trial.trial_id, {"loss": 0.42, "accuracy": 0.91})
print(completed.score_vector)  # scalarized score
print(completed.metrics)       # {"loss": 0.42, "accuracy": 0.91}
```

Extra fields beyond what your objectives require are stored in
the trial as `metrics` and can be inspected later.

!!! note
    For infeasible trials (where a metric exceeds its TLP limit), the corresponding entries in `.scores` and `.score_vector` are `float('inf')`. You can check for this with `math.isinf()`.

An exact replay of an accepted `tell` succeeds without another completion.
This permits retrying after an uncertain network response. Reporting different
metrics for the same completed ID fails (`ObjectiveError` locally,
`RemoteError` remotely). Completed history and the bounded receipt cache
provide replay information; an old evicted trial can eventually leave both.

### `study.cancel(trial_id) -> None`

Cancel pending work in either mode when an evaluation is abandoned. Its ID
remains consumed, and cancellation frees a pending slot under `max_trials`.
Completed trials cannot be cancelled.

### `remote.heartbeat(trial_id) -> int`

Renew a remote trial's lease and return its deadline as Unix milliseconds.
Use this for custom ask/tell workers with evaluations that might exceed the
server's lease. Calling it on a local study raises `ConfigurationError`.
The `run()` method manages renewal automatically on servers with heartbeat
support.

## The `run()` Convenience Method

For simple workflows, `study.run()` automates the ask/tell loop.

```python
study = Study(
    space=Space(x=Real(0.0, 1.0)),
    objectives=[Minimize("loss")],
)

def objective(params):
    return {"loss": train_model(params)}

study.run(objective, n_trials=100)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | callable | required | Function that takes a params dict and returns a metrics dict |
| `n_trials` | `int` | required | Number of trials to run |
| `n_workers` | `int` | `1` | Parallel workers: `<=1` = sequential, `N` = N parallel threads |

`run()` returns `self`, so you can chain.

```python
best = study.run(objective, n_trials=100).top_k(1)[0]
```

### Parallel Evaluation

With `n_workers > 1`,
`run()` dispatches trials concurrently using Python's
`ThreadPoolExecutor`. It keeps at most `n_workers` evaluations in flight,
processes each result as soon as that evaluation finishes, and immediately
replenishes the free slot. A slow early trial therefore does not hold up faster
later results, and exceptions cancel any still-pending trials before the pool
is shut down.
Remote `run()` renews each evaluation's lease while the callback runs, including
parallel evaluations, and stops renewal after completion or cancellation.
Transient renewal failures are retried within the last confirmed lease. If
the lease expires and the server rejects the result, `run()` raises
`RemoteError`; it does not silently count that evaluation as completed.
Older servers without a heartbeat endpoint retain their previous behavior.

```python
# Use 4 parallel workers
study.run(objective, n_trials=100, n_workers=4)

# Sequential (no thread pool overhead)
study.run(objective, n_trials=100, n_workers=1)
```

When using Python multiprocessing, select a `spawn` context and construct each
`Study` inside its worker. Forking a process after HOLA has initialized its
threaded native runtime can hang during `run()` or a later refit, even if simple
`ask()` and `tell()` calls succeed. For example, pass
`mp_context=multiprocessing.get_context("spawn")` to `ProcessPoolExecutor`,
and create the pool under `if __name__ == "__main__":`. The benchmark runners
already use spawned workers.

## Inspecting Results

All index fields (`trial_id`, `rank`, `pareto_front`) are
0-indexed.

### `study.top_k(k) -> list[CompletedTrial]`

Returns the top `k` trials found so far, as a list of
`CompletedTrial` objects. Returns an empty list if no trials
have been completed.

```python
top = study.top_k(1)
if top:
    best = top[0]
    print(best.score_vector)  # scalarized score
    print(best.params)        # {"x": 0.73}
    print(best.trial_id)      # 17
    print(best.metrics)       # original metrics dict
    print(best.scores)        # per-objective scores
    print(best.rank)          # rank in leaderboard
    print(best.completed_at)  # completion timestamp
```

### `study.trial_count() -> int`

Returns the number of completed trials.

```python
print(f"Completed {study.trial_count()} trials")
```

### `study.trials(sorted_by="index", include_infeasible=True) -> list[CompletedTrial]`

Returns retained completed trials as `CompletedTrial` objects. Each has
`.trial_id`, `.params`, `.score_vector`, `.scores`, `.metrics`,
`.rank`, `.pareto_front`, and `.completed_at`. Useful for
plotting convergence traces or custom analysis.

```python
# Compute running-best convergence trace
import math

best_so_far = float("inf")
trace = []
for trial in study.trials():
    sv = trial.score_vector  # dict of {objective group: scalarized score}
    obs = sum(sv.values()) if sv else float("inf")
    if math.isfinite(obs):
        best_so_far = min(best_so_far, obs)
    trace.append(best_so_far)
```

### `study.pareto_front(front=0, include_infeasible=False) -> list[CompletedTrial]`

Returns the Pareto front (non-dominated trials) for
multi-objective studies, specifically those with objectives
assigned to distinct groups. Each element is a `CompletedTrial`
with `.trial_id`, `.params`, `.scores`, `.metrics`, etc. The
`front` parameter is 0-indexed: `front=0` returns the first
(best) Pareto front, `front=1` returns the second front, and so
on. The `.pareto_front` field on each `CompletedTrial` is also
0-indexed.

```python
study = Study(
    space=Space(x=Real(0.0, 1.0)),
    objectives=[
        Minimize("loss", target=0.0, limit=5.0, priority=1.0, group="quality"),
        Minimize("latency", target=0.0, limit=100.0, priority=1.0, group="cost"),
    ],
    seed=42,
)
study.run(objective, n_trials=200, n_workers=1)

for trial in study.pareto_front():
    print(trial.scores)  # {"loss": 0.3, "latency": 42.0}
```

Returns an empty list for single-group (scalar) studies.

### `study.update_objectives(objectives) -> None`

Replace the objective definitions in either mode. The engine recomputes scores
and rankings for retained completed metrics and refits GMM selection as needed.
The parameter space and historical metrics are preserved.

## Saving and Resuming

### `study.save(path) -> None`

Save a local study's full checkpoint, including its configuration, strategy,
retained completed history, pending allocations, and retry state. Missing parent
directories are created. `save()` on a remote client raises `ConfigurationError`;
use the server's [checkpoint endpoint](rest-api.md) for a server-side file.

### `Study.load(path) -> Study`

Restore a local study from a full checkpoint and resume its ask/tell sequence.
Loading does not restart a previously hosted server. Failed saves or loads
raise `CheckpointError`.

```python
study.save("study.json")
restored = Study.load("study.json")
```

## Choosing a Strategy

Pass a string shortcut for defaults, or a strategy configuration
class for fine-grained control.

```python
# String shortcut (default settings)
Study(strategy="gmm", ...)

# Configuration class (custom settings)
Study(strategy=Gmm(refit_interval=10, elite_fraction=0.1), ...)
```

### GMM (default)

Gaussian Mixture Model strategy. Uses Sobol exploration followed
by GMM exploitation. Refits a GMM to the top `elite_fraction`
(default 12.5%) of eligible retained candidates every `refit_interval` (default 20)
completed trials. With multiple objective groups, elites are ordered
by non-domination rank and then descending crowding distance. The
exploration budget counts issued `ask` suggestions, including pending
trials. If concurrent asks reach that boundary before any empirical fit is
installed, HOLA continues the Sobol' sequence rather than sampling the
uninformed GMM prior. Uses the
[HOLA algorithm](concepts.md#gmm-strategy).
When `exploration_budget` is omitted, HOLA doubles
`min(floor(S / 5), 50 + 2n)` and then rounds down to a power of two, for budget
`S` and dimension `n`. `S` is `max_trials`, or 200 when no cap is set; that
fallback does not impose a trial cap. Thus an uncapped default study warms up
for 64 issued suggestions, and exploitation also waits for a successful
empirical fit.
GMM exploitation uses seeded Owen-scrambled Gauss–Sobol' points: one
Sobol' coordinate selects the component, and inverse-normal coordinates
sample within it. Each successfully installed GMM starts a new
epoch-specific scramble at its first point.

- Best for larger budgets (50+ trials) where exploration can
  transition to exploitation
- Concentrates samples in promising regions after warmup

```python
# Default GMM - equivalent to strategy="gmm"
Study(strategy=Gmm(), ...)

# Customized: refit more often, use top 10% of trials
Study(strategy=Gmm(refit_interval=10, elite_fraction=0.1), ...)
```

**`Gmm` parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `refit_interval` | `int` or `None` | 20 | How often the GMM is refit, in completed trials |
| `elite_fraction` | `float` or `None` | 0.125 | Fraction of top trials used for refitting. Must be in (0, 1]. |
| `exploration_budget` | `int` or `None` | auto | Number of issued Sobol exploration suggestions before GMM exploitation begins. Pending asks count against this budget. When omitted, HOLA doubles `min(floor(S/5), 50 + 2n)` and then rounds down to a power of two, for total budget `S` and dimension `n`; `S=200` when `max_trials` is unset. |
| `ongoing_exploration_period` | `int` or `None` | 0 | Continue global Sobol' exploration every Nth post-warmup suggestion. The default `0` disables it; explicit periods must be at least 2. |
| `max_components` | `int` or `None` | 1 | Maximum fitted mixture components. The effective count can be lower when the elite set is small. |
| `min_elite_samples` | `int` or `None` | 5 | Minimum feasible elite workset required before fitting. Must not exceed `max_refit_samples`. |
| `max_refit_samples` | `int` or `None` | 4096 | Maximum elite samples used by one GMM fit. Must be at least 1. |
| `max_refit_candidates` | `int` or `None` | 16384 | Maximum retained trials ranked during elite selection. Must be at least `max_refit_samples`; longer histories use deterministic stratified coverage. |

For a local study, `study.strategy_diagnostics()` returns counters that show
whether a fitted model has actually supplied suggestions. `gmm_fit_epoch` is
the number of installed empirical fits, `gmm_sampling_ready` indicates that a
fitted model can be sampled, `gmm_origin_suggestions` counts cumulative
suggestions from fitted models, and `issued_suggestions` counts all asks. A
missing GMM counter is `None`, including when an older checkpoint cannot
establish its history. This method raises `ConfigurationError` for remote
connections.

### Sobol

Owen-scrambled Sobol sequences provide quasi-random sampling with
better coverage than pure random. Good for initial exploration
and moderate-budget optimizations.

- Deterministic given a seed
- Fills the space more evenly than random sampling
- Works well for up to ~100--200 trials in moderate dimensions

```python
Study(strategy="sobol", ...)   # or
Study(strategy=Sobol(), ...)
```

### Random

Uniform pseudo-random sampling. A simple baseline.

- Deterministic given a seed
- No spatial structure; samples are independent.

```python
Study(strategy="random", ...)   # or
Study(strategy=Random(), ...)
```

## Going Distributed

### Hosting a Server

You can start a REST server directly from a local `Study`. It binds to
`127.0.0.1` by default, so clients on the same machine can connect. Set `host`
and an `auth_token` explicitly for network access, and terminate TLS at a
trusted reverse proxy when traffic leaves the host.

```python
study = Study(space=space, objectives=objectives)

# Blocking - serves until interrupted (Ctrl+C)
study.serve(port=8000)

# Background - returns after binding succeeds; study remains usable
study.serve(port=8000, background=True)
study.run(objective, n_trials=100)  # runs locally while server is active
study.stop()  # release the port when finished
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | `int` | `8000` | TCP port to listen on |
| `background` | `bool` | `False` | If `True`, returns after startup succeeds and keeps the server running in the shared runtime |
| `dashboard_path` | `str` or `None` | `None` | Path to a dashboard directory to serve the bundled UI. When omitted, no dashboard is served. Use `str(dashboard_dir())` to serve the bundled dashboard. |
| `host` | `str` | `"127.0.0.1"` | Keyword-only interface to bind; non-loopback hosting requires `auth_token` |
| `auth_token` | `str` or `None` | `None` | Keyword-only bearer token protecting API requests |
| `lease_seconds` | `float` | `7200.0` | Keyword-only lease duration for remote allocations; must be finite and represent at least one nanosecond, up to 3,153,600,000 seconds (100 years of 365 days) |

When `background=True`, the study continues to work locally. Both
local calls and remote HTTP requests share the same engine state,
so trials from any source appear in the same leaderboard.
Startup errors are raised to the caller, including an occupied port. One study
can own one background server at a time; call `study.stop()` before starting
another. `stop()` waits for graceful shutdown and is harmless when no server is
running. Hosting and stopping servers are available only on local studies.
Dropping the local study also requests shutdown of its background server.

```python
import os

study.serve(
    host="0.0.0.0", port=8000, background=True,
    auth_token=os.environ["HOLA_TOKEN"],
)
```

### Study.connect()

Connect to a running HOLA server (started via `study.serve()`,
`hola serve`, or any other means) using `Study.connect()`. The
returned object exposes the same methods as a local `Study`, but
forwards all calls as HTTP requests. The server holds the
leaderboard and strategy state.

```python
from hola_opt import Study

remote = Study.connect("http://localhost:8000")

# The same ask/tell/top_k interface as Study
trial = remote.ask()
remote.tell(trial.trial_id, {"loss": 0.42})
top = remote.top_k(1)

# Convenience method - automates the ask/tell loop
remote.run(my_function, n_trials=100, n_workers=4)

# Inspect results
print(remote.trial_count())    # number of completed trials
for t in remote.trials():      # all trials in insertion order
    print(t.trial_id, t.score_vector)

# Multi-objective: Pareto front
for t in remote.pareto_front():
    print(t.scores)
```

Remote requests use a 10-second connection timeout and a 30-second
whole-request timeout by default. Configured values must be finite, positive,
and representable as a duration of at least one nanosecond. The portable upper
bound is 3,153,600,000 seconds (100 years of 365 days); values beyond it raise
`ConfigurationError` before an HTTP client is created. A bearer token is
sent with every endpoint when provided:

```python
import os

remote = Study.connect(
    "https://hola.example.com",
    token=os.environ["HOLA_TOKEN"],
    connect_timeout=5.0,
    request_timeout=60.0,
)
```

Switching from local to distributed is mostly **replacing**
`Study(...)` **with** `Study.connect("http://...")` (you no
longer pass `space` / `objectives` here, since the server
already has them configured). All inspection methods (`top_k()`,
`trial_count()`, `trials()`, `pareto_front()`, and `run()`) work
on both modes. See the [Overview](#overview) for a comparison of
the two modes.

For the wire format, see the
[REST API Reference](rest-api.md).

## Examples

The `hola-py/examples/` directory contains complete runnable
examples:

| Example | Description |
|---------|-------------|
| `basic_optimization.py` | Minimizes 1D Forrester and 2D Branin functions. Shows both `study.run()` and the manual ask/tell loop. |
| `categorical_demo.py` | Mixed space with `Categorical`, `Real` (log10 scale), and `Integer` parameters. Simulates an optimizer hyperparameter search. |
| `gmm_explore_exploit.py` | Compares Sobol vs GMM strategies on Branin and Rastrigin. Shows how GMM concentrates samples after warmup. |
| `ml_hyperparameters.py` | Tunes a scikit-learn `GradientBoostingRegressor` with `Real`, log-scale `Real`, `Integer`, and `Categorical` parameters. Requires `scikit-learn`. |
| `multi_objective.py` | Optimizes error vs latency with TLP scoring and priority groups. Demonstrates Pareto-front optimization via `study.pareto_front()`. |

Run an example.

```bash
uv run --directory hola-py python examples/basic_optimization.py
```

Run that command from the repository root. The `--directory` option
selects the `hola-py` project and its virtual environment explicitly.
