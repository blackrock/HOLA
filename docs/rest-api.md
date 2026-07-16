# REST API Reference

The HOLA server exposes a REST API for distributed
optimization. All endpoints accept and return JSON. Browser
cross-origin access is disabled by default; allow trusted origins
explicitly with the repeatable `hola serve --cors-origin <ORIGIN>`
option. CLI, Python, and other non-browser clients are not subject
to CORS.

Most users do not need to call the REST API directly.
`Study.connect()` in Python and `hola worker` in the CLI
handle HTTP communication automatically. This reference is
for custom integrations and debugging.

## Base URL

```
http://localhost:8000
```

The port is configurable via `hola serve --port <PORT>`.

## Authentication

By default, a local HOLA server does not require authentication.
When the server is started with `--auth-token <TOKEN>`, all API
endpoints require this header:

```http
Authorization: Bearer <TOKEN>
```

Operators may configure a separate `--read-token` for dashboards and
monitoring. That role can use GET endpoints, SSE, and metrics, but receives HTTP
401 from ask, tell, cancel, heartbeat, objective updates, and checkpoint saves.

This includes mutation endpoints, read-only endpoints, and the
`GET /api/events` SSE stream. Operators can deliberately leave reads
open with `--allow-unauthenticated-reads`; this is intended only for
trusted networks because study parameters, metrics, and events may be
sensitive.
The CLI requires an auth token when binding the server to a non-local
host.

## Error Format

All error responses return a stable machine-readable `code` plus a
human-readable `error` message.

```json
{"code": "tell_failed", "error": "Trial 42 has already been completed with different metrics"}
```

---

## Endpoints

### POST /api/ask

Request the next trial to evaluate.

**Request.** No body required. Distributed workers should attach a unique
`Idempotency-Key` header and reuse that key until they receive a valid response.
If a connection fails after the server allocated a trial, retrying with the same
key returns that trial instead of allocating duplicate work. Keys must contain
1–128 ASCII characters. The built-in `hola worker` command does this
automatically.

Server allocations carry a lease (two hours by default). Complete or cancel the
trial before it expires, or renew it with `POST /api/heartbeat`. Expired work is
reclaimed lazily and a late tell is rejected deterministically.

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
curl -X POST http://localhost:8000/api/ask \
  -H "Idempotency-Key: $(uuidgen)"
```

---

### POST /api/heartbeat

Renew the lease for a pending distributed trial. This is a mutation endpoint
and requires the bearer token when authentication is configured.

**Request**

```json
{"trial_id": 0}
```

**Response (200)**

```json
{"status":"ok","trial_id":0,"lease_expires_at_ms":1783735200000}
```

The expiry is an absolute Unix timestamp in milliseconds. A missing,
completed, cancelled, or already-expired trial returns
`{"code":"heartbeat_failed", ...}` with HTTP 400.

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

Strict JSON has no numeric representation for non-finite floats. Objective
metric values may therefore use the exact string sentinels `"inf"`, `"-inf"`,
or `"nan"`; other strings are ordinary raw values and do not parse as numbers.

**Response (200)**

```json
{
  "status": "ok",
  "trial_count": 1,
  "trial": {
    "trial_id": 0,
    "params": {
      "learning_rate": 0.00316,
      "num_layers": 5,
      "optimizer": "adam",
      "momentum": 0.85
    },
    "score_vector": {"loss": 0.42},
    "scores": {"loss": 0.42},
    "metrics": {
      "loss": 0.42,
      "latency": 120.5
    },
    "rank": 0,
    "pareto_front": 0,
    "completed_at": 1736935800
  },
  "post_commit_warnings": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` on success |
| `trial_count` | integer | Total number of completed trials after this tell |
| `trial` | object | Newly completed trial created by this `tell` |
| `post_commit_warnings` | array of strings | Refit/checkpoint maintenance failures that occurred after the trial committed. The tell remains successful; clients should surface these warnings and must not retry it as an ingestion failure. |

The returned `trial.trial_id` matches the `trial_id` in the
request.

An exact duplicate request replays the retained successful receipt without
committing again or emitting a duplicate event. **Error (400)** is returned if
the trial ID is unknown/cancelled or if a completed ID is retried with different
metrics.

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
| `k` | integer | required | Number of top trials to return |

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
| `score_vector` | object | Mapping of priority-group names to aggregated scores. Non-finite values use lossless string sentinels: `"inf"`, `"-inf"`, and `"nan"`. |
| `scores` | object | Per-objective scores, with the same `"inf"`, `"-inf"`, and `"nan"` sentinels. |
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

Get the Pareto front (the set of non-dominated trials) for
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
| `score_vector` | object | Mapping of priority-group names to aggregated scores. Non-finite values use lossless string sentinels: `"inf"`, `"-inf"`, and `"nan"`. |
| `scores` | object | Per-objective scores, with the same `"inf"`, `"-inf"`, and `"nan"` sentinels. |
| `metrics` | object | Original metrics dict |
| `rank` | integer | 0-indexed overall rank |
| `pareto_front` | integer | 0-indexed Pareto front index |
| `completed_at` | integer | Unix timestamp in seconds |

For scalar studies, this endpoint returns an empty array because
there are no Pareto fronts to report.

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
| `sorted_by` | string | `"index"` | Sort order: `"index"` (insertion order) or `"rank"` |
| `include_infeasible` | boolean | true | Whether to include infeasible trials |

**Response (200)**

```json
[
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
]
```

The response is an array of completed trials. Each trial has the
following fields.

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | Unique trial identifier |
| `params` | object | Parameter values |
| `score_vector` | object | Mapping of priority-group names to aggregated scores. Non-finite values use lossless string sentinels: `"inf"`, `"-inf"`, and `"nan"`. |
| `scores` | object | Per-objective scores, with the same `"inf"`, `"-inf"`, and `"nan"` sentinels. |
| `metrics` | object | Original metrics dict |
| `rank` | integer | 0-indexed overall rank |
| `pareto_front` | integer | 0-indexed Pareto front index |
| `completed_at` | integer | Unix timestamp in seconds |

**Example**

```bash
curl http://localhost:8000/api/trials
```

---

### GET /api/trial/{trial_id}

Get one completed trial by public trial ID.

**Path parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trial_id` | integer | Public trial ID returned by `ask` and `tell` |

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_infeasible` | boolean | true | Whether to return infeasible completed trials |

**Response (200)**

```json
{
  "trial_id": 17,
  "params": {"learning_rate": 0.001, "num_layers": 4},
  "score_vector": {"loss": 0.42},
  "scores": {"loss": 0.42},
  "metrics": {"loss": 0.42, "latency": 95.3},
  "rank": 0,
  "pareto_front": 0,
  "completed_at": 1736935800
}
```

**Error (404).** Returned if no completed trial exists for the
given ID, or if the trial is infeasible and `include_infeasible`
is false.

```json
{"error": "Trial 17 not found"}
```

**Example**

```bash
curl http://localhost:8000/api/trial/17
```

---

### GET /api/trial/{trial_id}/status

Get the authoritative lifecycle state used to reconcile a distributed worker.
This read follows the server's read-authentication policy. Unlike the ranked
single-trial endpoint, `completed` also recognizes a retained completion
receipt after `max_leaderboard_size` has evicted the trial from the leaderboard.

**Response (200)**

```json
{"status":"ok","trial_id":17,"state":"completed"}
```

`state` is one of the following.

| State | Meaning |
|-------|---------|
| `completed` | The exact trial is in the leaderboard or its recent completion receipt is retained. |
| `pending` | The exact trial is still live and may be cancelled. |
| `not_pending` | The trial is expired, cancelled, unknown, or no longer has a retained completion receipt; it must not be cancelled again. |

The response always echoes the requested `trial_id`, so clients can reject a
mismatched or malformed acknowledgement before taking lifecycle action.

**Example**

```bash
curl http://localhost:8000/api/trial/17/status
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
| `target` | float | no | TLP target value |
| `limit` | float | no | TLP limit value |
| `priority` | float | no | TLP score at the limit (when configured) and relative weight (default: 1.0) |

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
      "type": "real",
      "min": 0.0001,
      "max": 0.1,
      "scale": "log10"
    },
    {
      "name": "num_layers",
      "type": "integer",
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

Save the current server state as a full JSON checkpoint file.

**Request**

```json
{
  "description": "After 100 trials"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | no | Optional description stored in the checkpoint metadata |

**Response (200)**

```json
{
  "status": "ok",
  "checkpoint_type": "full",
  "path": "./checkpoints/checkpoint_1783684800000_000000.json",
  "trials_saved": 100,
  "created_at": 1783684800
}
```

The saved file includes completed trials, strategy state, and study
configuration. The server always generates a unique filename beneath its
operator-configured `--checkpoint-dir`; clients cannot select or replace a
server-side path. The returned `path` identifies the generated file.

**Error (400)**

```json
{"code": "invalid_request", "error": "Failed to deserialize the JSON body..."}
```

**Error (500)**

```json
{"code": "checkpoint_save_failed", "error": "failed to save checkpoint"}
```

**Example**

```bash
curl -X POST http://localhost:8000/api/checkpoint/save \
  -H "Content-Type: application/json" \
  -d '{"description": "After 100 trials"}'
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

### GET /api/event_cursor

Get the current SSE replay watermark before fetching a REST snapshot.
The cursor is a decimal string so JavaScript clients retain the full `u64`
value without precision loss.

```json
{"last_event_id": "42"}
```

To avoid losing a completion between `GET /api/trials` and opening the event
stream, read this cursor first, fetch the snapshot, then send it as the
`Last-Event-ID` header on `GET /api/events`. Events that overlap the snapshot
may be delivered twice; clients should upsert trials by `trial_id`.

---

### GET /api/events

We stream server-sent events (SSE) for real-time updates.
The dashboard uses this endpoint for live monitoring.

**Response.** SSE stream with monotonic `id` fields and `data` fields containing
JSON. Reconnect with `Last-Event-ID` to replay retained events. A
`stream_reset` or `stream_lagged` event means the client must refresh its REST
snapshot before continuing.

**Event types**

**TrialCompleted.** Emitted after each successful `tell`.

```json
{
  "type": "TrialCompleted",
  "trial_id": 42,
  "score": 0.42,
  "trial": {
    "trial_id": 42,
    "params": {"learning_rate": 0.001},
    "score_vector": {"loss": 0.42},
    "scores": {"loss": 0.42},
    "metrics": {"loss": 0.42},
    "rank": 0,
    "pareto_front": 0,
    "completed_at": 1736935800
  }
}
```

**ObjectivesChanged.** Emitted after an objective update commits. Because the
change can alter every historical score, rank, and Pareto front, clients must
perform the cursor-safe full REST resynchronization described above.

```json
{"type": "ObjectivesChanged"}
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
# {"status":"ok","trial_count":1,"trial":{"trial_id":0,...}}

# Check the top trial
curl -s http://localhost:8000/api/top_k?k=1
# [{"trial_id":0,"params":{"learning_rate":0.00316,...},"score_vector":{"loss":0.42},...}]

# View all trials
curl -s http://localhost:8000/api/trials | jq .
```
