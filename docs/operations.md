# Operations Runbook

This runbook covers a long-running HOLA server and distributed workers. HOLA
binds to loopback by default. Any non-loopback bind requires an API token, and
that token protects reads, writes, metrics, and the event stream unless the
operator explicitly enables unauthenticated reads.

## Start securely

Keep the tokens outside shell history, configuration files, and the process
command line. The server reads the write token from `HOLA_API_TOKEN` and the
read-only token from `HOLA_READ_TOKEN`:

```bash
export HOLA_API_TOKEN="$(openssl rand -hex 32)"
export HOLA_READ_TOKEN="$(openssl rand -hex 32)"
hola serve study.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --checkpoint-dir ./checkpoints \
  --cors-origin https://optimizer.example.com
```

Do not pass a token as the value of `--auth-token`, `--read-token`, or a
worker's `--token`: a flag value is visible to every local user in the
process list for as long as the process runs. Where environment variables
are unsuitable, store each token in a file readable only by the service
account and use `--auth-token-file`, `--read-token-file`, and the worker's
`--token-file`.

The server speaks plain HTTP and has no TLS of its own. Terminate TLS at a
trusted reverse proxy whenever traffic leaves the host, and give workers and
Python clients the proxy's `https://` URL. They refuse to send a token to a
non-loopback `http://` address unless started with `--allow-insecure-http`
(`allow_insecure_http=True` in Python), which is appropriate only on a network
where the token may be read in transit. They also never follow redirects, so
configure the final URL. Do not use `--allow-unauthenticated-reads` when
parameters, metrics, or study progress are sensitive.

A server started without a token binds to loopback and only answers requests
whose `Host` header names `localhost` or a loopback IP address. Other host
names receive HTTP 403 (`host_forbidden`), which keeps web pages from reaching
a local server through DNS rebinding. A reverse proxy that forwards its public
host name therefore needs a token-protected server.

Files under `--dashboard` are served to anyone who can reach the server,
without a token, because the UI has to load before a token can be entered. The
server refuses to start when that directory contains `--checkpoint-dir` or the
study's automatic checkpoint directory. Keep only the dashboard's own files in
it.

## Health and readiness

The unauthenticated probe endpoints do not expose study data:

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/readyz
```

`/healthz` proves the HTTP task is alive. `/readyz` additionally reports that
the router and engine state are available. Remove an instance from traffic when
either probe fails repeatedly.

## Metrics and logs

Scrape Prometheus text from `/api/metrics`; use the read-only token so the
monitoring system cannot allocate trials or change study state:

```bash
curl --fail \
  -H "Authorization: Bearer $HOLA_READ_TOKEN" \
  http://127.0.0.1:8000/api/metrics
```

The endpoint reports monotonic completed trials, currently retained trials,
pending trials, HTTP request/failure/latency totals, manual and unattended
auto-checkpoint/rotation failures, failed strategy refits, and published events.
Alert when pending trials grow continuously without completions, failure
counters rise, a worker's durable tell outbox is nonempty for an extended
period, or readiness fails. A refit failure does not roll back the already
committed trial or objective update; inspect the warning log and repair the
model/data issue. Before its first empirical model, GMM maintenance retries on
the next completion; subsequent models use the configured refit cadence.
HOLA emits structured HTTP tracing at INFO level with the `x-request-id` in its
request span and response. Preserve that ID in proxy logs when correlating a
worker error with the server.

Stable JSON error responses have a machine-readable `code` and a human-readable
`error`. Operators should group failures by `code`; the message may gain detail
between releases.

## Worker recovery

Use `hola worker --mode exec` when possible. It validates command output and
persists a successful result to its server-scoped outbox before sending it.
After a crash or uncertain network response, restart with the same
`--outbox-dir`; queued tells are replayed before new work is requested. Exact
duplicate tells are accepted, while conflicting metrics are rejected. Queued
tells are sent with the worker's token, so on Unix the worker requires the
outbox to be a real directory (not a symbolic link) owned by its own user or
by root and not writable by anyone else, and refuses to start otherwise.

Workers reuse an `Idempotency-Key` while retrying an uncertain ask, so a lost
response does not allocate another trial. Ask keys and lease deadlines are
included in full checkpoints. Failed or timed-out commands are terminated as a
process tree and their trials are cancelled. Custom long-running workers can
renew work through `POST /api/heartbeat`; otherwise the server reclaims an
expired lease on the next job-lifecycle or metrics request. The server also caps
pending work, so both memory use and orphan lifetime are bounded.
Python `Study.connect(...).run()` renews leases during evaluations automatically.
Custom Python ask/tell loops must call `remote.heartbeat(trial_id)` themselves
when work can outlast its lease.

## Checkpoints and restart

Full checkpoints contain configuration, strategy state, completed history,
pending trials, cancelled IDs, and the next trial ID. Save one before planned
maintenance:

```bash
curl --fail -X POST \
  -H "Authorization: Bearer $HOLA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"description":"pre-maintenance"}' \
  http://127.0.0.1:8000/api/checkpoint/save
```

The server generates a unique filename beneath `--checkpoint-dir` and returns
it as `path`; clients cannot choose a destination. Copy or back up that file
only after the request succeeds. On restart, load that checkpoint through the
study configuration. Pending IDs remain valid, so late workers can report their
results without ID reuse. A checkpoint with an unsupported format, incompatible
study configuration, or inconsistent counts is rejected before live state
changes.

Only one save runs at a time; a second request receives HTTP 409
(`checkpoint_in_progress`). A save that has started finishes even if the
request times out, and the server log then records the `path` it wrote. The
server keeps at most 32 saved files (`--max-saved-checkpoints`) and never
deletes one itself: when the limit is reached, further saves receive HTTP 409
(`checkpoint_quota_exceeded`) until you archive or remove older
`checkpoint_<time>_<sequence>.json` files. Automatic checkpoints configured in
the study are rotated separately by `max_checkpoints`.

Checkpoints contain the full study state. On Linux and macOS, HOLA creates
checkpoint files readable and writable only by the server's user (mode 0600),
and creates missing checkpoint directories with mode 0700. A directory that
already exists keeps its permissions, so to share checkpoints with a group,
create the directory yourself and adjust the files you want to share. On
Windows the files inherit the access control of their directory.

A checkpoint larger than 512 MiB cannot be loaded, so a save that would exceed
that size fails without writing anything. Bound the retained history with
`max_leaderboard_size` if a study approaches it.

## Resource limits

The server bounds what a client can make it hold or compute; the
[REST API reference](rest-api.md#limits) lists every limit. The ones that
matter for capacity planning:

- **Connections.** At most 1024 connections are open at once; further ones are
  closed as they arrive. A client has 120 seconds to send a complete request
  head, and an idle keep-alive connection is closed after the same time. Put a
  reverse proxy in front of the server when it faces a network you do not
  control: the proxy should terminate TLS, limit connections per client, and
  keep its upstream keep-alive timeout below 120 seconds.
- **Event streams.** At most 64 `/api/events` streams are open at once. Each
  dashboard tab uses one.
- **Ranked reads.** `top_k`, `pareto_front`, `trials`, and single-trial reads
  rank the retained history. Four run at a time on a separate thread pool, so
  large reads queue behind each other instead of delaying tells and probes.
- **Retained history.** The server keeps every completed trial and its
  reported metrics unless the study sets `max_leaderboard_size`. Memory,
  ranking time, and checkpoint size all grow with that history, and any client
  holding the write token can add to it. Set `max_trials` to stop at a budget
  and `max_leaderboard_size` to cap what is retained.
- **Dimensions.** The `gmm` and `auto` strategies accept at most 1024
  parameters, because each mixture component stores a dense covariance matrix.
  Use `random` for larger spaces.

## Graceful shutdown

Send `SIGTERM` (or Ctrl-C interactively). The server stops accepting new
connections, closes SSE event streams when shutdown begins, and drains its Axum
shutdown path for up to ten seconds. The bounded deadline closes any other
remaining long-lived connection so shutdown cannot hang indefinitely. HOLA does
not invent a checkpoint destination during shutdown; for planned maintenance,
complete the checkpoint request above before sending the signal. For an
unexpected termination, resume from the latest successful periodic or manual
checkpoint and restart workers with their original outbox directories.

## Incident checklist

1. Record the response `x-request-id`, error `code`, worker trial ID, and current
   checkpoint filename.
2. Check `/healthz`, `/readyz`, and `/api/metrics` locally.
3. Preserve worker outbox files; do not delete them to clear an alert.
4. Stop new workers if pending work is rising without completions.
5. Save a checkpoint if the server is responsive, then restart it and replay
   worker outboxes.
6. Treat a conflicting duplicate tell as a data-integrity incident: preserve
   both metric payloads and do not overwrite the accepted result.
