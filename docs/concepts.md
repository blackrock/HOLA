# Concepts

This page explains the core ideas behind HOLA's optimization
pipeline. For API specifics, see the [Python Guide](python-guide.md),
[CLI Guide](cli-guide.md), or [REST API Reference](rest-api.md).

## The Ask/Tell Paradigm

We separate **suggestion** from **evaluation**.

1. **Ask.** The engine suggests a set of parameter values to try.
2. **Evaluate.** Your code runs with those parameters and produces
   metrics.
3. **Tell.** You report the metrics back to the engine.

This decoupling allows the same engine to work locally (Python
`Study`), over a network (REST API / `Study.connect()`), or in a
distributed setting with a server and multiple workers.

```mermaid
sequenceDiagram
    Engine->>Your Code: ask() → params {lr: .., ...}
    Your Code->>Engine: tell() → metrics {loss: .., ...}
```

## Architecture

We organize HOLA into three layers. The **engine core** is a Rust
optimization library that handles spaces, strategies, scales, and
the leaderboard. The **orchestration layer** wraps the engine
core behind a JSON-based interface, resolving spaces and strategies
at runtime from configuration and scalarizing metrics against the
study objectives. The **public
interfaces** (Python bindings, CLI, and REST server) all delegate
to the orchestration layer, so users never interact with Rust
internals directly.

For day-to-day work we recommend the Python API. The Rust internals
are an implementation detail you do not need to think about.

## Data Flow

We route every trial through the same pipeline.

```
ask()
  ├─ Strategy proposes candidate in [0,1]^n (unit hypercube)
  ├─ Space maps [0,1]^n → domain values (e.g., lr=0.001, layers=5)
  └─ Returns JSON params dict to caller

evaluate (your code)
  └─ Returns JSON metrics dict (e.g., {"loss": 0.42, "latency": 120})

tell()
  ├─ Engine validates metrics against the objective schema
  ├─ Engine computes priority-weighted costs within explicit groups
  ├─ Leaderboard stores the trial (params, score_vector, metrics, timestamp)
  └─ Strategy refits to scalar-ranked or Pareto/NSGA-II-ranked elites
```

## The Unit Hypercube

All strategies operate internally in the **unit hypercube**
$[0,1]^n$, where $n$ is the number of dimensions. We define a
bijection between $[0,1]^n$ and the actual domain.

- `to_unit_cube(domain_value)` $\to [0,1]$ normalizes a domain
  value.
- `from_unit_cube(unit_value)` $\to$ `domain_value` denormalizes
  back to the domain.

This standardization means strategies never need to know about
parameter scales or types. They always propose uniform vectors in
$[0,1]^n$, and the space handles the mapping.

## Scale Transformations

We support three scales for real-valued parameters.

**Linear** (default). Uniform sampling in the domain. Use this for
parameters where all values in the range are equally plausible.

```python
Real(0.0, 1.0)  # samples uniformly from [0, 1]
```

**Log10.** Uniform sampling in $\log_{10}$ space. Use this for
parameters spanning orders of magnitude. Learning rates are a
classic example.

```python
Real(1e-4, 0.1, scale="log10")  # samples uniformly from 10^(-4) to 10^(-1)
```

A value of 0.5 in the unit hypercube maps to
$10^{-2.5} \approx 0.00316$, not $0.05$.

**Log.** Uniform sampling in natural log space. Available in Python
as `Real(a, b, scale="log")`.

## Search Strategies

### GMM Strategy (default) {#gmm-strategy}

The default strategy is the
[Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model).
We fit a GMM to the **top quantile** of completed trials, then
sample new candidates from the fitted model. This concentrates
samples in regions where good results have been observed.
For multiple objective groups, the elite order uses non-domination
rank followed by descending crowding distance.

The lifecycle follows three phases.

1. **Warmup.** The first trials build up the leaderboard. HOLA continues
   Sobol' sampling until the first empirical fit is installed.
2. **Refit.** Every `refit_interval` trials (default 20), we refit
   the GMM to the top 25% of trials, subject to the configured minimum
   feasible elite workset.
3. **Exploit.** New samples are drawn from the updated GMM,
   focusing on promising regions. By default, every fifth post-warmup
   suggestion remains a global Sobol' exploration point.

GMM exploitation uses seeded Owen-scrambled Gauss–Sobol' points.
One Sobol' coordinate selects the mixture component, and inverse-normal
coordinates sample within that component. Each successfully installed GMM
starts a new epoch-specific scramble at its first point, giving every fixed
model a balanced sequence prefix without repeating the same scramble across
refits.

The exploration budget counts issued suggestions from `ask`,
including suggestions that are still pending. GMM refits are based
on completed trials in the leaderboard. Pending asks that cross the nominal
warmup boundary before the first completed-data fit remain Sobol' suggestions;
the unfitted prior is never used as exploitation.

`max_components` bounds mixture complexity (default 3), while
`min_elite_samples` can delay fitting until a feasible elite workset reaches a
requested size (default 1). The effective component count may be smaller: the
implementation requires enough elite samples to support each component.

Two implementation limits keep refitting bounded on unusually long studies.
At most `max_refit_samples` elite points enter one fit (default 4096), and at
most `max_refit_candidates` retained trials are ranked to choose them (default
16384). Histories within the candidate limit are ranked globally. Longer
histories use deterministic chronological strata spanning the full retained
history, rather than a newest-only window. These are implementation safeguards;
they do not change the abstract definition of the elite set. Scalar selection
is linear in the candidate count. General multi-group non-dominated sorting is
quadratic in the worst case, so long-running studies with many groups may use a
smaller `max_refit_candidates` value.

This strategy works well for larger budgets (50+ trials) where you
want to transition from exploration to exploitation. The more
trials you run, the more the GMM focuses on the best regions.

### Random Strategy

We draw uniform pseudo-random samples in $[0,1]^n$. The sequence
is deterministic given a seed (auto-incremented per sample).

This strategy works well for baselines, very low-dimensional
spaces, or when you need independent samples.

### Sobol Strategy

We use
[Owen-scrambled Sobol sequences](https://en.wikipedia.org/wiki/Sobol_sequence)
(Burley's 2020 variant) for quasi-random sampling. Sobol sequences
fill the space more evenly than pseudo-random sampling; the
distance between any two points is more uniform.

This strategy works well for initial exploration, moderate budgets
(up to approximately 200 trials), and when even coverage of the
space matters.

## Objective Scalarization

We convert each metric field into a directed, priority-weighted contribution.
Contributions in the same priority group are summed. A study with one group
uses that scalar group cost for ranking; multiple groups retain a cost vector
and use Pareto/NSGA-II ranking. GMM refits use that same ordering to select
multi-group elites.

### Single Field

With `Minimize("loss")`, the score is the value of the
`"loss"` field. With `Maximize("accuracy")`, we negate the value
since the engine always minimizes internally.

### Weighted Sum Within a Group

When several objectives share one `group`, we compute that group's
priority-weighted sum

$$
\text{score} = \sum_i p_i d_i x_i
$$

where $p_i$ is the priority, $d_i$ is $+1$ for minimize and $-1$
for maximize, and $x_i$ is the field value.

### Target-Limit-Priority (TLP)

TLP objectives provide fine-grained control over multi-objective
scalarization. We assign three parameters to each objective field.

- **Target ($t$).** The "satisfactory" value. At or better than
  target, the contribution is 0.
- **Limit ($l$).** The worst acceptable boundary. At the limit, the
  contribution is $p$; crossing beyond it makes the trial infeasible and the
  contribution infinite.
- **Priority ($p$).** The contribution's score at the limit and its relative
  weight within a group. The slope between target and limit is
  $p/(l-t)$; priority is not itself the slope unless $l-t=1$.

Between target and limit, we interpolate linearly

$$
\text{contribution} = p \cdot \frac{x - t}{l - t}
$$

For **minimize** objectives, $t < l$ (e.g., loss target = 0.01,
limit = 1.0). For **maximize** objectives, $t > l$ (e.g.,
accuracy target = 0.95, limit = 0.5). The declared objective `type` and
this ordering must agree: a configuration whose ordering contradicts its
`type` (for example `type: maximize` with `target < limit`) is rejected
at validation rather than silently optimized in the wrong direction.

Within each priority group, the final group cost is the sum of its field
contributions

$$
\text{group cost} = \sum_{i \in \text{group}} \text{contribution}_i
$$

Trials where any field exceeds its limit are effectively infeasible (the
corresponding group cost is $\infty$). When `group` is omitted, each field uses
its own group, so two or more omitted-group objectives form Pareto axes rather
than one unconditional weighted sum.

## The Leaderboard

By default we retain all completed trials. Long-running studies can set
`max_leaderboard_size` to retain a bounded recent/ranked working set while the
monotonic completed count continues to drive budgets, refits, and checkpoints.
The leaderboard provides the following.

**Lazy ranking.** We rank trials on demand, not on every insert.

**`top_k(k)`.** Returns the $k$ best trials.

**`top_quantile(q)`.** Returns trials in the top $q$ fraction
(used by GMM refitting).

**Pareto front.** For multi-objective studies (objectives with
distinct `group` labels), we provide `pareto_front()`,
`non_dominated_sort()`, and NSGA-II crowding distance.
Two-group front ranking uses an exact $O(N \log N)$ sweep. Exact ranking for
three or more groups remains quadratic in the retained population; configure a
finite `max_leaderboard_size` for large many-objective studies so latency and
memory have an explicit bound.

**Rescalarization.** When objectives are updated mid-run, we
rescalarize all existing trials with the new objectives.

For multi-objective studies the engine uses a vector leaderboard
that stores per-group TLP scores, enabling Pareto dominance
queries. For single-objective studies we use a scalar leaderboard
for efficient ranking. The choice is made automatically based on
the number of distinct groups in the objectives.

Each trial record contains the following fields.

| Field | Description |
|-------|-------------|
| `trial_id` | Unique integer identifier |
| `params` | The parameter values (domain space) |
| `score_vector` | The scalarized score(s) |
| `scores` | Per-objective score dict |
| `metrics` | The original metrics dict from `tell()` |
| `rank` | Trial rank in the leaderboard |
| `pareto_front` | Pareto front index (multi-objective only) |
| `completed_at` | When the trial was completed |

## Persistence

We support atomic JSON checkpoints for both warm starts and exact
resumes.

- **Leaderboard-only checkpoint.** Completed trials with params, scores,
  metrics, and timestamps. This legacy format is a warm-start artifact.
- **Full checkpoint.** A leaderboard checkpoint plus study configuration,
  search strategy state (for example, Sobol sequence position or GMM
  parameters), pending and cancelled work, leases, idempotency records,
  completion receipts, and the next trial ID.

Checkpoints enable the following.

- Resuming an optimization after a crash or restart
- Offline analysis in the dashboard
- Carrying over a leaderboard to a new engine (warm-start)

Loading a current full checkpoint restores its runtime state, including
pending IDs, so workers can report results issued before the restart without
ID reuse. Loading a legacy leaderboard-only checkpoint necessarily invalidates
outstanding jobs because the file does not identify them. HOLA then begins a
fresh trial-ID epoch and reconciles the configured strategy with imported
history by advancing sampling counters and refitting a model where applicable.
That legacy path is a warm start, not an exact continuation.

We write checkpoint files atomically (first to a temp file, then
rename) to prevent corruption.
