# REST API Reference

The HOLA server exposes a REST API for distributed
optimization. All endpoints accept and return JSON. We enable
permissive CORS by default.

Most users do not need to call the REST API directly.
`Study.connect()` in Python and `hola worker` in the CLI
handle HTTP communication automatically. This reference is
for custom integrations and debugging.

## Base URL

```
http://localhost:8000
```

The port is configurable via `hola serve --port <PORT>`.

## Error Format

All error responses return a JSON object with an `error`
field.

```json
{"error": "Trial 42 has already been completed"}
```

---

## Endpoints

### POST /api/ask

Request the next trial to evaluate.

**Request.** No body required.

**Response (200)**

```json
{
  "trial_id": 0,
  "params": {
    "learning_rate": 0.00316,
    "num_layers": 5,
    "optimizer": "adam",
    "momentum": 0.85
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | Unique trial identifier |
| `params` | object | Parameter name-value pairs in the domain space |

**Example**

```bash
curl -X POST http://localhost:8000/api/ask
```

---

### POST /api/tell

Report the result of a trial.

**Request**

```json
{
  "trial_id": 0,
  "metrics": {
    "loss": 0.42,
    "latency": 120.5
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | The trial ID from a previous `ask` |
| `metrics` | object | Key-value pairs of metric results. Must include fields referenced by objectives. |

**Response (200)**

```json
{
  "status": "ok",
  "trial_count": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` on success |
| `trial_count` | integer | Total number of completed trials after this tell |

**Error (400).** Returned if the trial ID is unknown or has
already been told.

```json
{"error": "Trial 0 has already been completed"}
```

**Example**

```bash
curl -X POST http://localhost:8000/api/tell \
  -H "Content-Type: application/json" \
  -d '{"trial_id": 0, "metrics": {"loss": 0.42, "latency": 120.5}}'
```

---

### GET /api/top_k

Get the top k trials found so far.

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | integer | 1 | Number of top trials to return |

**Response (200)**

```json
[
  {
    "trial_id": 17,
    "params": {
      "learning_rate": 0.001,
      "num_layers": 4,
      "optimizer": "adam",
      "momentum": 0.9
    },
    "score_vector": {"loss": 0.42},
    "scores": {"loss": 0.42},
    "metrics": {
      "loss": 0.42,
      "latency": 95.3
    },
    "rank": 0,
    "pareto_front": 0,
    "completed_at": 1736935800
  }
]
```

Each element in the array has the following fields.

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | Unique trial identifier |
| `params` | object | Parameter values |
| `score_vector` | object | Mapping of priority-group names to aggregated scores. Infeasible values appear as the string `"inf"`. |
| `scores` | object | Per-objective scores. Infeasible values appear as the string `"inf"`. |
| `metrics` | object | Original metrics dict from `tell` |
| `rank` | integer | 0-indexed overall rank |
| `pareto_front` | integer | 0-indexed Pareto front index |
| `completed_at` | integer | Unix timestamp in seconds |

We return an empty array if no trials have been completed.

**Example**

```bash
curl http://localhost:8000/api/top_k?k=5
```

---

### GET /api/pareto_front

Get the Pareto front---the set of non-dominated trials---for
multi-objective studies.

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `front` | integer | 0 | Pareto front index (0 = first/best front) |
| `include_infeasible` | boolean | false | Whether to include infeasible trials |

**Response (200)**

```json
[
  {
    "trial_id": 3,
    "params": {
      "learning_rate": 0.005,
      "num_layers": 4
    },
    "score_vector": {"loss": 0.31, "cost": 85.2},
    "scores": {
      "loss": 0.31,
      "latency": 85.2
    },
    "metrics": {
      "loss": 0.31,
      "latency": 85.2
    },
    "rank": 0,
    "pareto_front": 0,
    "completed_at": 1736935800
  },
  {
    "trial_id": 7,
    "params": {
      "learning_rate": 0.001,
      "num_layers": 6
    },
    "score_vector": {"loss": 0.42, "cost": 52.1},
    "scores": {
      "loss": 0.42,
      "latency": 52.1
    },
    "metrics": {
      "loss": 0.42,
      "latency": 52.1
    },
    "rank": 1,
    "pareto_front": 0,
    "completed_at": 1736935805
  }
]
```

Each element in the array has the following fields.

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | Unique trial identifier |
| `params` | object | Parameter values |
| `score_vector` | object | Mapping of priority-group names to aggregated scores. Infeasible values appear as the string `"inf"`. |
| `scores` | object | Per-objective scores. Infeasible values appear as the string `"inf"`. |
| `metrics` | object | Original metrics dict |
| `rank` | integer | 0-indexed overall rank |
| `pareto_front` | integer | 0-indexed Pareto front index |
| `completed_at` | integer | Unix timestamp in seconds |

**Error (400).** Returned for single-group (scalar) studies.

```json
"pareto_front() is only available for multi-objective studies (objectives with distinct groups)"
```

**Example**

```bash
curl http://localhost:8000/api/pareto_front
```

---

### GET /api/trials

Get all completed trials.

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sorted_by` | string | `"index"` | Sort order---`"index"` (insertion order) or `"rank"` |
| `include_infeasible` | boolean | true | Whether to include infeasible trials |

**Response (200)**

```json
{
  "trials": [
    {
      "trial_id": 0,
      "params": {"learning_rate": 0.01, "num_layers": 3},
      "score_vector": {"loss": 0.85},
      "scores": {"loss": 0.85},
      "metrics": {"loss": 0.85, "latency": 50.2},
      "rank": 1,
      "pareto_front": 0,
      "completed_at": 1736935800
    },
    {
      "trial_id": 1,
      "params": {"learning_rate": 0.001, "num_layers": 7},
      "score_vector": {"loss": 0.42},
      "scores": {"loss": 0.42},
      "metrics": {"loss": 0.42, "latency": 120.5},
      "rank": 0,
      "pareto_front": 0,
      "completed_at": 1736935805
    }
  ],
  "total": 2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trials` | array | All completed trials |
| `total` | integer | Total number of completed trials |

Each trial in the array has the following fields.

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | Unique trial identifier |
| `params` | object | Parameter values |
| `score_vector` | object | Mapping of priority-group names to aggregated scores. Infeasible values appear as the string `"inf"`. |
| `scores` | object | Per-objective scores. Infeasible values appear as the string `"inf"`. |
| `metrics` | object | Original metrics dict |
| `rank` | integer | 0-indexed overall rank |
| `pareto_front` | integer | 0-indexed Pareto front index |
| `completed_at` | integer | Unix timestamp in seconds |

**Example**

```bash
curl http://localhost:8000/api/trials
```

---

### GET /api/objectives

Get the current objective configuration.

**Response (200)**

```json
{
  "objectives": [
    {
      "field": "loss",
      "obj_type": "minimize",
      "target": null,
      "limit": null,
      "priority": 1.0,
      "group": null
    },
    {
      "field": "latency",
      "obj_type": "minimize",
      "target": 100.0,
      "limit": 500.0,
      "priority": 0.5,
      "group": "cost"
    }
  ]
}
```

**Example**

```bash
curl http://localhost:8000/api/objectives
```

---

### PATCH /api/objectives

Update objectives mid-run. We rescalarize all existing trials
with the new objectives.

**Request**

```json
{
  "objectives": [
    {
      "field": "accuracy",
      "obj_type": "maximize",
      "priority": 1.0
    }
  ]
}
```

Each objective in the array has the following fields.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `field` | string | yes | Metrics field name |
| `obj_type` | string | yes | `"minimize"` or `"maximize"` |
| `priority` | float | no | Relative weight (default: 1.0) |
| `target` | float | no | TLP target value |
| `limit` | float | no | TLP limit value |

**Response (200)**

```json
{
  "status": "ok",
  "rescalarized_trials": 42
}
```

| Field | Type | Description |
|-------|------|-------------|
| `rescalarized_trials` | integer | Number of trials rescalarized with the new objectives |

**Example**

```bash
curl -X PATCH http://localhost:8000/api/objectives \
  -H "Content-Type: application/json" \
  -d '{"objectives": [{"field": "accuracy", "obj_type": "maximize", "priority": 1.0}]}'
```

---

### GET /api/space

Get parameter space metadata.

**Response (200)**

```json
{
  "params": [
    {
      "name": "learning_rate",
      "type": "continuous",
      "min": -4.0,
      "max": -1.0,
      "scale": "log10"
    },
    {
      "name": "num_layers",
      "type": "discrete",
      "min": 1,
      "max": 10,
      "scale": null
    },
    {
      "name": "optimizer",
      "type": "categorical",
      "min": 0.0,
      "max": 2.0,
      "scale": "linear",
      "choices": ["adam", "sgd", "rmsprop"]
    }
  ]
}
```

**Example**

```bash
curl http://localhost:8000/api/space
```

---

### POST /api/checkpoint/save

Save the current state as a JSON checkpoint file.

**Request**

```json
{
  "path": "/tmp/checkpoint.json",
  "description": "After 100 trials"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | no | File path for the checkpoint (default: `"checkpoint.json"`) |
| `description` | string | no | Optional description stored in the checkpoint metadata |

**Response (200)**

```json
{
  "status": "ok",
  "path": "/tmp/checkpoint.json",
  "trials_saved": 100
}
```

**Error (500)**

```json
{"error": "Permission denied: /read-only/path.json"}
```

**Example**

```bash
curl -X POST http://localhost:8000/api/checkpoint/save \
  -H "Content-Type: application/json" \
  -d '{"path": "checkpoint_100.json", "description": "After 100 trials"}'
```

---

### POST /api/cancel

Cancel a pending trial that has been asked but not yet told.

**Request**

```json
{
  "trial_id": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | The trial ID to cancel |

**Response (200)**

```json
{"status": "ok"}
```

**Error (400)**

```json
{"error": "Trial 5 is not pending"}
```

**Example**

```bash
curl -X POST http://localhost:8000/api/cancel \
  -H "Content-Type: application/json" \
  -d '{"trial_id": 5}'
```

---

### GET /api/trial_count

Get the number of completed trials.

**Response (200)**

```json
{"trial_count": 42}
```

**Example**

```bash
curl http://localhost:8000/api/trial_count
```

---

### GET /api/events

We stream server-sent events (SSE) for real-time updates.
The dashboard uses this endpoint for live monitoring.

**Response.** SSE stream with `data` fields containing JSON.

**Event types**

**TrialCompleted.** Emitted after each successful `tell`.

```json
{"type": "TrialCompleted", "trial_id": 42, "score": 0.42}
```

**RefitOccurred.** Emitted when the GMM strategy is refit.

```json
{"type": "RefitOccurred", "n_trials": 40}
```

**Example**

```bash
curl -N http://localhost:8000/api/events
```

---

## Complete Example: Ask/Tell Loop with curl

```bash
# Start a server (in another terminal)
# hola serve config.yaml --port 8000

# Ask for a trial
TRIAL=$(curl -s -X POST http://localhost:8000/api/ask)
echo "$TRIAL"
# {"trial_id":0,"params":{"learning_rate":0.00316,"num_layers":5,"optimizer":"adam","momentum":0.85}}

# Extract trial ID
TRIAL_ID=$(echo "$TRIAL" | jq '.trial_id')

# Tell the result
curl -s -X POST http://localhost:8000/api/tell \
  -H "Content-Type: application/json" \
  -d "{\"trial_id\": $TRIAL_ID, \"metrics\": {\"loss\": 0.42, \"latency\": 120}}"
# {"status":"ok","trial_count":1}

# Check the top trial
curl -s http://localhost:8000/api/top_k?k=1
# [{"trial_id":0,"params":{"learning_rate":0.00316,...},"score_vector":{"loss":0.42},...}]

# View all trials
curl -s http://localhost:8000/api/trials | jq .
```
