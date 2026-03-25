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

Both modes expose the same methods (`ask`, `tell`, `top_k`, …).
You pick one based on **process layout**, not on different math.

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
| `Gmm` | GMM strategy configuration (refit_interval, elite_fraction, exploration_budget) |
| `Sobol` | Sobol strategy configuration |
| `Random` | Random strategy configuration |

All classes are imported from the `hola` module.

```python
from hola_opt import (
    Study, Space, Trial, CompletedTrial,
    Real, Integer, Categorical,
    Minimize, Maximize,
    Gmm, Sobol, Random,
)
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

HOLA scalarizes multiple objectives into a single score using a
priority-weighted sum. By default, all objectives have
`priority=1.0`.

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
- **limit.** The "unacceptable" value. Trials at or beyond limit
  score infinity (effectively infeasible).
- **priority.** Per-objective weight/slope ($P_i$) in the TLP
  formula:
  $\varphi_i = P_i \times (\text{value} - \text{target}) / (\text{limit} - \text{target})$.

Between target and limit, the score is interpolated linearly and
scaled by `priority`. See
[Concepts: TLP Scalarization](concepts.md#target-limit-priority-tlp)
for the full explanation.

### Priority Groups

To enable Pareto-front multi-objective optimization, assign
objectives to different groups using the `group` parameter.
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
| `max_trials` | `int` or `None` | `None` | Maximum number of trials. When set, `ask()` raises after this many trials have been dispatched. |

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

Reports the result of a trial. `metrics` must be a dict
containing at least the fields specified in your objectives.
Returns a `CompletedTrial`.

```python
completed = study.tell(trial.trial_id, {"loss": 0.42, "accuracy": 0.91})
print(completed.score_vector)  # scalarized score
print(completed.metrics)       # {"loss": 0.42, "accuracy": 0.91}
```

Extra fields beyond what your objectives require are stored in
the trial as `metrics` and can be inspected later.

!!! note
    For infeasible trials (where a metric exceeds its TLP limit), the corresponding entries in `.scores` and `.score_vector` are `float('inf')`. You can check for this with `math.isinf()`.

!!! warning
    Each trial ID can only be told once. Calling `tell` with the same ID twice raises a `ValueError`.

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
| `n_workers` | `int` | `0` | Parallel workers: `0` = auto-detect CPU count, `1` = sequential, `N` = N parallel threads |

`run()` returns `self`, so you can chain.

```python
best = study.run(objective, n_trials=100).top_k(1)[0]
```

### Parallel Evaluation

With `n_workers > 1` (or the default `0` for auto-detect),
`run()` dispatches trials concurrently using Python's
`ThreadPoolExecutor`. Each batch asks for `n_workers` trials,
evaluates them in parallel, then tells the results.

```python
# Use 4 parallel workers
study.run(objective, n_trials=100, n_workers=4)

# Sequential (no thread pool overhead)
study.run(objective, n_trials=100, n_workers=1)
```

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

Returns all trials as `CompletedTrial` objects. Each has
`.trial_id`, `.params`, `.score_vector`, `.scores`, `.metrics`,
`.rank`, `.pareto_front`, and `.completed_at`. Useful for
plotting convergence traces or custom analysis.

```python
# Compute running-best convergence trace
import math

best_so_far = float("inf")
trace = []
for trial in study.trials():
    sv = trial.score_vector
    obs = sv if isinstance(sv, (int, float)) else sv[0] if sv else float("inf")
    if isinstance(obs, (int, float)) and math.isfinite(obs):
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
(default 25%) of trials every `refit_interval` (default 20)
completed trials. Uses the
[HOLA algorithm](concepts.md#gmm-strategy).

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
| `elite_fraction` | `float` or `None` | 0.25 | Fraction of top trials used for refitting. Must be in (0, 1]. |
| `exploration_budget` | `int` or `None` | auto | Number of Sobol exploration trials before GMM exploitation begins. When omitted, computed automatically from the number of dimensions. |

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

You can start a REST server directly from a local `Study`,
making it accessible to remote workers.

```python
study = Study(space=space, objectives=objectives)

# Blocking - serves until interrupted (Ctrl+C)
study.serve(port=8000)

# Background - serves in a background thread, study remains usable
study.serve(port=8000, background=True)
study.run(objective, n_trials=100)  # runs locally while server is active
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | `int` | `8000` | TCP port to listen on |
| `background` | `bool` | `False` | If `True`, runs in a background thread and returns immediately |

When `background=True`, the study continues to work locally. Both
local calls and remote HTTP requests share the same engine state,
so trials from any source appear in the same leaderboard.

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
uv run python hola-py/examples/basic_optimization.py
```

Run from the repository root. If your working directory is
`hola-py/`, use `uv run python examples/basic_optimization.py`
instead.
