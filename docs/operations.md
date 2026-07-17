# Operations Runbook

This runbook covers a long-running HOLA server and distributed workers. HOLA
binds to loopback by default. Any non-loopback bind requires an API token, and
that token protects reads, writes, metrics, and the event stream unless the
operator explicitly enables unauthenticated reads.

## Start securely

Keep the token outside shell history and configuration files:

```bash
export HOLA_API_TOKEN="$(openssl rand -hex 32)"
export HOLA_READ_TOKEN="$(openssl rand -hex 32)"
hola serve study.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --checkpoint-dir ./checkpoints \
  --auth-token "$HOLA_API_TOKEN" \
  --read-token "$HOLA_READ_TOKEN" \
  --cors-origin https://optimizer.example.com
```

Terminate TLS at a trusted reverse proxy when traffic leaves the host. Do not
use `--allow-unauthenticated-reads` when parameters, metrics, or study progress
are sensitive.

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
duplicate tells are accepted, while conflicting metrics are rejected.

Workers reuse an `Idempotency-Key` while retrying an uncertain ask, so a lost
response does not allocate another trial. Ask keys and lease deadlines are
included in full checkpoints. Failed or timed-out commands are terminated as a
process tree and their trials are cancelled. Custom long-running workers can
renew work through `POST /api/heartbeat`; otherwise the server reclaims an
expired lease on the next job-lifecycle or metrics request. The server also caps
pending work, so both memory use and orphan lifetime are bounded.

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
