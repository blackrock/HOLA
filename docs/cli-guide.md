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
`POST /api/tell`. If the script exits with non-zero status,
the worker cancels the trial automatically.

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
| `priority` | no | Relative weight (default: 1.0) |
| `target` | no | The "satisfactory" value (for TLP) |
| `limit` | no | The "unacceptable" value (for TLP) |
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
```

| Field | Default | Description |
|-------|---------|-------------|
| `type` | `"gmm"` | Strategy type: `"gmm"`, `"sobol"`, or `"random"` |
| `refit_interval` | `20` | How often the GMM refits (only used by `"gmm"`) |
| `seed` | none | Seed for reproducible runs. When omitted, Sobol uses 42, others use random seeds. |
| `exploration_budget` | none | Number of issued Sobol exploration suggestions before switching to GMM exploitation. Pending asks count against this budget. When omitted, we use a formula based on `total_budget`. |
| `elite_fraction` | `0.25` | Fraction of top trials used for GMM refitting. Must be in (0.0, 1.0]. |

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
| `--auth-token` | none | Bearer token required for write-capable API endpoints |
| `--checkpoint-dir` | checkpoint config directory or config file directory | Directory where dashboard/API checkpoint saves are allowed |
| `--cors-origin` | none | Allowed browser CORS origin. Repeat for multiple origins |

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

### Callback mode (default)

In callback mode, the worker loop works as follows.

1. `POST /api/ask` to get a trial from the server
2. Run the `--exec` command via `sh -c` with three
   environment variables set
   - `HOLA_SERVER`. The server URL
     (e.g., `http://localhost:8000`).
   - `HOLA_TRIAL_ID`. The numeric trial ID.
   - `HOLA_PARAMS`. Trial parameters as a JSON string.
3. The script is responsible for calling `POST /api/tell` to
   report results back to the server
4. If the script exits with non-zero status, the worker
   cancels the trial via `POST /api/cancel`
5. Repeat

If the server is unreachable, the worker retries every
5 seconds.

### Exec mode

With `--mode exec`, the worker runs the command and handles
reporting on its behalf.

1. `POST /api/ask` to get a trial
2. Run the `--exec` command with `HOLA_PARAMS` set
3. Parse the command's stdout as a JSON metrics object
4. `POST /api/tell` to report the result
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

Each invocation of the `--exec` command runs in its own
`sh -c` process, so `HOLA_PARAMS` (along with `HOLA_SERVER`
and `HOLA_TRIAL_ID`) is per-process. Multiple concurrent
workers are safe: each worker's script sees only its own
trial's parameters.

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

#### Python (callback mode, hola Python client)

If you have the `hola` Python package installed, you can use
`Study.connect()` for a nicer API.

```python
#!/usr/bin/env python3
# train.py - worker script using the hola Python client
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

While the server is running, open `dashboard/index.html` in a
browser and enter the server URL (e.g.,
`http://localhost:8000`). The dashboard connects via SSE and
shows live convergence plots, a trial table, and Pareto
scatter.

See the [Dashboard Guide](dashboard.md) for details.

## Checkpointing

### What gets saved

A leaderboard checkpoint saves all completed trials: their
parameters, metrics, and scores. A full checkpoint
additionally saves strategy state (e.g. GMM model parameters).
We use JSON as the checkpoint format.

### Automatic checkpointing

We configure automatic checkpointing in YAML.

```yaml
checkpoint:
  directory: ./checkpoints
  interval: 50
  max_checkpoints: 5
```

This saves a checkpoint every 50 completed trials, keeping
the 5 most recent. Automatic checkpoints are leaderboard
checkpoints: they preserve completed trials and can be used to
warm-start a restarted server.

### Manual checkpointing

We can save a checkpoint at any time via the REST API.

```bash
curl -X POST http://localhost:8000/api/checkpoint/save \
  -H "Content-Type: application/json" \
  -d '{"path": "my_checkpoint.json"}'
```

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

On startup, the server loads the specified checkpoint file. Full
checkpoints restore both the leaderboard and search strategy state.
Legacy leaderboard-only checkpoints are still accepted as a
warm-start path; they restore completed trials but do not restore
strategy state.

Checkpoint loads intentionally clear any pending or cancelled
in-flight trials from the engine state. Restored studies resume from
the completed trial history, and the next `ask` receives a fresh ID
after the restored completed trials.
