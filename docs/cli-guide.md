# CLI & Distributed Usage

For multi-machine or language-agnostic deployments, we
provide a CLI that runs an optimization server and dispatches
work to any number of workers. For single-machine use, the
Python `Study` class is simpler; see the
[Python Guide](python-guide.md).

The `hola` CLI has two subcommands.

**`hola serve`.** Starts an HTTP optimization server from a
YAML config file. It exposes the [REST API](rest-api.md) that
workers and the Python `Study.connect()` client use to request
trials and report results. We optionally serve the dashboard
UI from a local directory with the `--dashboard` flag.

**`hola worker`.** A loop that polls the server for trials,
executes your command for each one, and handles trial
lifecycle. By default it uses callback mode, where the script
is responsible for reporting results back to the server via
`POST /api/tell`. If the script exits with non-zero status, the worker cancels
the trial only when the server still confirms it is pending; an
already-completed tell is authoritative.

## YAML Configuration

The server reads a YAML file that defines the parameter space,
objectives, strategy, and optional checkpointing.

### Full Example

```yaml
# example_study.yaml

space:
  learning_rate:
    type: real
    min: 0.0001
    max: 0.1
    scale: log10

  num_layers:
    type: integer
    min: 1
    max: 10

  optimizer:
    type: categorical
    choices:
      - adam
      - sgd
      - rmsprop
      - adamw

  momentum:
    type: real
    min: 0.5
    max: 0.99

objectives:
  - field: loss
    type: minimize
    priority: 1.0
    group: quality
  - field: latency
    type: minimize
    target: 100
    limit: 500
    priority: 0.5
    group: cost

strategy:
  type: gmm
  refit_interval: 20
  seed: 42

# Optional: automatic checkpointing
# checkpoint:
#   directory: ./checkpoints
#   interval: 50
#   max_checkpoints: 5
```

### Space Configuration

Each parameter in the `space:` section has a `type` and
type-specific fields.

#### Real

```yaml
temperature:
  type: real
  min: 0.0
  max: 2.0
```

We default to `"linear"` scale; you can also set `"log10"` or
`"log"`.

#### Log10 scale

```yaml
learning_rate:
  type: real
  min: 0.0001
  max: 0.1
  scale: log10
```

We specify `min` and `max` as actual values, not exponents.
This matches the Python API where
`Real(1e-4, 0.1, scale="log10")` also takes actual values.
Internally, HOLA samples uniformly in log10 space.

#### Integer

```yaml
num_layers:
  type: integer
  min: 1
  max: 10
```

Integer values from min to max, inclusive.

#### Categorical

```yaml
optimizer:
  type: categorical
  choices:
    - adam
    - sgd
    - rmsprop
```

### Objectives Configuration

We require at least one objective. Each objective has the
following fields.

| Field | Required | Description |
|-------|----------|-------------|
| `field` | yes | The metrics field name to optimize |
| `type` | yes | `"minimize"` or `"maximize"` |
| `target` | no | The "satisfactory" value (for TLP) |
| `limit` | no | The worst acceptable boundary (for TLP) |
| `priority` | no | TLP score at the limit (when configured) and relative weight (default: 1.0) |
| `group` | no | Priority group name. Objectives in the same group are summed; distinct groups form Pareto axes. Omit for single-group (scalar) studies. |

```yaml
objectives:
  - field: loss
    type: minimize
    priority: 1.0
    group: quality
  - field: latency
    type: minimize
    target: 100
    limit: 500
    priority: 0.5
    group: cost
```

See [Concepts: TLP Scalarization](concepts.md#target-limit-priority-tlp)
for details on target/limit semantics.

### Strategy Configuration

```yaml
strategy:
  type: gmm               # "gmm" (default), "sobol", or "random"
  refit_interval: 20       # how often GMM refits (used by "gmm")
  seed: 42                 # optional seed for reproducible runs
  exploration_budget: 50   # number of Sobol asks before switching to GMM
  elite_fraction: 0.25     # fraction of top trials used for GMM fitting (default: 0.25)
  ongoing_exploration_period: 5  # every Nth post-warmup ask is Sobol (0 disables)
  max_components: 3        # maximum fitted GMM components
  min_elite_samples: 1     # minimum feasible elite workset before fitting
  max_refit_samples: 4096  # maximum elite samples passed to one GMM fit
  max_refit_candidates: 16384  # maximum trials ranked to choose elites
```

| Field | Default | Description |
|-------|---------|-------------|
| `type` | `"gmm"` | Strategy type: `"gmm"`, `"sobol"`, or `"random"` |
| `refit_interval` | `20` | How often the GMM refits (only used by `"gmm"`) |
| `seed` | none | Seed for reproducible runs. When omitted, HOLA draws one seed once and records it in full checkpoints. |
| `exploration_budget` | none | Number of issued Sobol exploration suggestions before switching to GMM exploitation. Pending asks count against this budget. When omitted, we use a formula based on `total_budget` and the search dimension. |
| `elite_fraction` | `0.25` | Fraction of top trials used for GMM refitting. Must be in (0.0, 1.0]. |
| `ongoing_exploration_period` | `5` | Continue global Sobol' exploration every Nth post-warmup suggestion. Use `0` to disable; explicit periods must be at least 2. |
| `max_components` | `3` | Maximum fitted GMM components. The effective count can be lower for small elite sets. |
| `min_elite_samples` | `1` | Minimum feasible elite workset required before fitting. Must not exceed `max_refit_samples`. |
| `max_refit_samples` | `4096` | Maximum elite samples passed to one GMM fit. Must be at least 1. |
| `max_refit_candidates` | `16384` | Maximum retained trials ranked to choose elites. Must be at least `max_refit_samples`; longer histories use deterministic stratified coverage of the full retained history. |

GMM exploitation uses seeded Owen-scrambled Gauss–Sobol' points. One Sobol'
coordinate selects the mixture component, and inverse-normal coordinates sample
within that component. Each successfully installed GMM starts a new
epoch-specific scramble at its first point.

### Checkpoint Configuration

```yaml
checkpoint:
  directory: ./checkpoints    # where to save checkpoint files
  interval: 50                # save every N trials
  max_checkpoints: 5          # keep only the N most recent
  load_from: ./checkpoints/checkpoint_000100.json  # resume from this checkpoint
```

## Starting a Server

```bash
hola serve config.yaml --port 8000
```

| Flag | Default | Description |
|------|---------|-------------|
| `config` | required | Path to the YAML configuration file |
| `--host` | `127.0.0.1` | Host/interface to bind. Use `0.0.0.0` explicitly for network access |
| `--port` | `8000` | Port to listen on |
| `--dashboard` | none | Path to a dashboard directory to serve at `/` (e.g. `--dashboard ./dashboard`) |
| `--auth-token` | none | Bearer token required for all API endpoints |
| `--read-token` | none | Optional read-only token for dashboards, SSE, and metrics; never permits mutations |
| `--allow-unauthenticated-reads` | off | Explicitly leave read-only endpoints and SSE open when a token is configured |
| `--checkpoint-dir` | checkpoint config directory or config file directory | Directory where dashboard/API checkpoint saves are allowed |
| `--cors-origin` | none | Allowed browser CORS origin. Repeat for multiple origins |
| `--lease-seconds` | `7200` | Time a distributed trial may remain pending without completion, cancellation, or heartbeat |

The server starts listening on `127.0.0.1:<port>` by default and exposes
the [REST API](rest-api.md). Binding a non-local host requires `--auth-token`
or the `HOLA_API_TOKEN` environment variable.

## Running Workers

```bash
hola worker --server http://localhost:8000 --exec "python train.py"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | required | URL of the HOLA server |
| `--exec` | required | Shell command to execute for each trial |
| `--mode` | `callback` | Worker mode: `"callback"` or `"exec"` |
| `--token` | none | Bearer token for servers started with `--auth-token` |
| `--request-timeout` | `30` | Maximum seconds for an HTTP request |
| `--command-timeout` | `3600` | Maximum seconds before the command process tree is terminated |
| `--outbox-dir` | `.hola-worker-outbox` | Durable, server-scoped queue for exec-mode tells |

### Callback mode (default)

In callback mode, the worker loop works as follows.

1. `POST /api/ask` to get a trial from the server
2. Run the `--exec` command through the platform shell (`sh -c` on Unix,
   `cmd /C` on Windows) with these
   environment variables set
   - `HOLA_SERVER`. The server URL
     (e.g., `http://localhost:8000`).
   - `HOLA_TRIAL_ID`. The numeric trial ID.
   - `HOLA_PARAMS`. Trial parameters as a JSON string.
   - `HOLA_API_TOKEN`. Set only when the worker has a token
     configured (via `--token` or the `HOLA_API_TOKEN`
     environment variable). When present, the script must send
     it as a `Bearer` header on its `POST /api/tell` and
     `POST /api/cancel` requests, since those write endpoints
     require authorization on a token-protected server.
3. The script is responsible for calling `POST /api/tell` to
   report results back to the server
4. Reconcile the exact trial lifecycle with the server. A completed tell stays
   authoritative even if a later heartbeat is rejected or the script exits
   non-zero. The worker calls `POST /api/cancel` only if the trial is still
   pending and the script failed or exited without completing it.
5. Repeat

If the server is unreachable, the worker retries with the same ask
idempotency key, so a lost response cannot allocate duplicate work.
Workers also remain compatible with servers that predate the lifecycle-status
and heartbeat endpoints: they fall back to the exact completed-trial lookup,
use a validated heartbeat when available, and rely on the older server's atomic
cancel operation when no lease protocol exists.

### Exec mode

With `--mode exec`, the worker runs the command and handles
reporting on its behalf.

1. `POST /api/ask` to get a trial
2. Run the `--exec` command with `HOLA_PARAMS` set
3. Require a successful exit and parse bounded stdout as a JSON metrics object
4. Persist the tell to the durable outbox, then `POST /api/tell`; uncertain
   responses are retried before any new work is requested
5. Repeat

```bash
hola worker --server http://localhost:8000 \
  --exec "python train.py" --mode exec
```

### The `HOLA_PARAMS` environment variable

We pass trial parameters to your command via the `HOLA_PARAMS`
environment variable as a JSON string.

```bash
HOLA_PARAMS='{"learning_rate": 0.001, "num_layers": 5, "optimizer": "adam", "momentum": 0.9}'
```

Each invocation of the `--exec` command runs in its own platform-shell
process group or Windows Job Object, so `HOLA_PARAMS` (along with `HOLA_SERVER`
and `HOLA_TRIAL_ID`) is per-process. Multiple concurrent
workers are safe: each worker's script sees only its own
trial's parameters. A command timeout terminates and reaps the whole process
tree rather than leaving descendants running.

### Worker Script Examples

#### Python (callback mode, stdlib only)

```python
#!/usr/bin/env python3
# train.py - worker script for HOLA (callback mode)
import json
import os
import urllib.request

# Read parameters and server info from environment
params = json.loads(os.environ["HOLA_PARAMS"])
server = os.environ["HOLA_SERVER"]
trial_id = os.environ["HOLA_TRIAL_ID"]

lr = params["learning_rate"]
layers = params["num_layers"]
optimizer = params["optimizer"]
momentum = params["momentum"]

# Your training code here
loss = train_model(lr=lr, layers=layers, optimizer=optimizer, momentum=momentum)
latency = measure_latency()

# Report results back to the server
payload = json.dumps({"trial_id": int(trial_id), "metrics": {"loss": loss, "latency": latency}})
req = urllib.request.Request(
    f"{server}/api/tell",
    data=payload.encode(),
    headers={"Content-Type": "application/json"},
)
urllib.request.urlopen(req)
```

#### Bash (callback mode, curl)

```bash
#!/bin/bash
# train.sh - worker script for HOLA (callback mode)

# Parse parameters with jq
LR=$(echo "$HOLA_PARAMS" | jq -r '.learning_rate')
LAYERS=$(echo "$HOLA_PARAMS" | jq -r '.num_layers')

# Run your training
LOSS=$(python train.py --lr "$LR" --layers "$LAYERS" 2>/dev/null)

# Report results back to the server
curl -s -X POST "$HOLA_SERVER/api/tell" \
  -H "Content-Type: application/json" \
  -d "{\"trial_id\": $HOLA_TRIAL_ID, \"metrics\": {\"loss\": $LOSS}}"
```

#### Python (callback mode, hola-opt Python client)

If you have the `hola-opt` Python package (imported as `hola_opt`)
installed, you can use `Study.connect()` for a nicer API.

```python
#!/usr/bin/env python3
# train.py - worker script using the hola-opt Python client
import json
import os
from hola_opt import Study

params = json.loads(os.environ["HOLA_PARAMS"])
trial_id = int(os.environ["HOLA_TRIAL_ID"])
remote = Study.connect(os.environ["HOLA_SERVER"])

# Your training code here
loss = train_model(**params)
latency = measure_latency()

# Report results
remote.tell(trial_id, {"loss": loss, "latency": latency})
```

#### Python (exec mode)

```python
#!/usr/bin/env python3
# train.py - worker script for HOLA (exec mode: stdout JSON)
import json
import os
import sys

params = json.loads(os.environ["HOLA_PARAMS"])

# Use stderr for logging (stdout must be pure JSON)
print("Training started...", file=sys.stderr)

loss = train_model(**params)
latency = measure_latency()

# Print metrics as JSON to stdout
print(json.dumps({"loss": loss, "latency": latency}))
```

!!! important
    In exec mode, the worker script must print **only** a
    JSON object to stdout. Any other stdout output will cause
    a parse error. Use stderr for logging.

## Multi-Machine Setup

To run distributed optimization across multiple machines, we
start a server on one machine and point workers at it from the
others.

**Machine A (server):**

```bash
hola serve config.yaml --host 0.0.0.0 --port 8000 --auth-token "$HOLA_API_TOKEN"
```

**Machines B, C, D (workers):**

```bash
hola worker --server http://machine-a:8000 --token "$HOLA_API_TOKEN" --exec "python train.py"
```

Each worker independently polls the server for trials. The
server handles concurrent ask/tell requests safely.

You can also connect from Python on any machine.

```python
from hola_opt import Study

remote = Study.connect("http://machine-a:8000")
trial = remote.ask()
# ... evaluate ...
remote.tell(trial.trial_id, metrics)

# Or use run() to automate the loop
remote.run(my_function, n_trials=50, n_workers=4)

# Inspect results from any machine
print(remote.trial_count())
for t in remote.pareto_front():  # multi-objective studies
    print(t.scores)
```

## Monitoring with the Dashboard

Start the server with `--dashboard ./dashboard`, then open its URL
(for example, `http://localhost:8000/`). Serving the UI and API from
HOLA keeps browser requests on the same origin. The dashboard connects
via SSE and shows live convergence plots, a trial table, and Pareto
scatter. If you host the UI elsewhere, allow that exact origin with
`--cors-origin`.

See the [Dashboard Guide](dashboard.md) for details.

## Checkpointing

### What gets saved

A leaderboard-only checkpoint saves completed trials: their parameters,
metrics, and scores. A current full checkpoint additionally saves study
configuration, strategy state (for example, Sobol position or GMM model
parameters), pending and cancelled work, leases, idempotency records,
completion receipts, and the next trial ID. We use JSON as the checkpoint
format.

### Automatic checkpointing

We configure automatic checkpointing in YAML.

```yaml
checkpoint:
  directory: ./checkpoints
  interval: 50
  max_checkpoints: 5
```

This saves a full checkpoint every 50 completed trials, keeping the 5 most
recent. It preserves runtime and strategy state for an exact server resume.

### Manual checkpointing

We can save a checkpoint at any time via the REST API.

```bash
curl -X POST http://localhost:8000/api/checkpoint/save \
  -H "Content-Type: application/json" \
  -d '{"description": "manual CLI checkpoint"}'
```

The server generates the filename beneath its configured checkpoint directory
and returns the full server-side path in the response.

Or from the dashboard's Checkpoints panel.
Manual REST and dashboard saves write full checkpoints with
completed trials, strategy state, and study configuration.

### Resuming from a checkpoint

Add the `load_from` field to your YAML checkpoint config.

```yaml
checkpoint:
  directory: ./checkpoints
  interval: 50
  max_checkpoints: 5
  load_from: ./checkpoints/checkpoint_000100.json
```

On startup, the server loads the specified checkpoint file. Current full
checkpoints restore the leaderboard, strategy, and runtime state. Pending IDs
remain valid, so a worker can finish work issued before the restart without the
ID being reused.

Legacy leaderboard-only checkpoints remain supported as a warm-start path.
Because they contain no runtime state, loading one invalidates all outstanding
jobs and starts a fresh trial-ID epoch so late results cannot collide with new
work. HOLA reconciles the configured strategy with the imported history by
advancing its sampling counters and, where applicable, refitting its model;
this is a history-informed warm start rather than an exact continuation.
