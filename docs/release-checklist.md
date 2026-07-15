# Release Verification Checklist

Run this checklist on the exact release commit after the CI and Security Audit
workflow runs for that commit are green. Record both run URLs and SHAs together
with the commit, date, operator, browser/OS versions, and measured timings in
the release notes. A failed item blocks the release; it is not a waiver.

## Hosted compatibility

- Confirm the Rust matrix passed on Linux, macOS, and Windows, including the
  Rust 1.87 MSRV and feature-minimal checks.
- Confirm Python 3.10, 3.12, and 3.14 tested the wheel built by the job rather
  than an editable source build.
- Confirm release-wheel smoke tests passed on Linux x86-64/aarch64, macOS
  x86-64/aarch64, and Windows x86-64.
- Download every CLI archive and run `hola --help` on each native platform.
- Confirm Cargo, pip, and npm audits report no unexpired vulnerability.

## Browser and accessibility

Use current Firefox, Chromium, and Safari (where available), once with network
access disabled after loading the repository checkout.

- Open `dashboard/index.html` and load a checkpoint without any CDN or network
  request. Confirm convergence, Pareto, parallel-coordinate, and table views
  remain usable.
- Load a mixed continuous/integer/categorical study. Confirm categorical axes
  use configured choice order and keyboard sorting cycles ascending,
  descending, and unsorted.
- Check min/min, min/max, max/max, tied, constrained, and all-infeasible
  objective fixtures. Infeasible trials must never be highlighted as Pareto
  optimal while a feasible trial exists.
- Connect to a live server, force an SSE reconnect, then load a local file while
  events are arriving. The view must converge without duplicates, gaps, or a
  stale live response overwriting offline state.
- Navigate all controls with the keyboard and inspect the page with a screen
  reader. Controls, canvases, connection state, sort direction, and chart text
  alternatives must be announced meaningfully.
- At 375 CSS pixels wide and at 200% zoom, no required control may be clipped or
  made unreachable. With reduced motion enabled, no essential state may depend
  on animation.

## Scale budgets

Run the ignored Rust probes in a debug build:

```bash
RUN_EXPENSIVE_BENCHMARKS=1 cargo test --locked -p opt_engine \
  --test integration leaderboard_scalability -- --ignored --nocapture
cargo test --locked -p opt_engine gmm_hot_path_throughput_probe \
  -- --ignored --nocapture
```

On a four-core hosted CI runner, each 100,000-trial scalar operation and the
two-objective Pareto front must finish within 5 seconds. The 100,000-suggestion
GMM probe and 2,000-sample fit must each finish within 5 seconds. Compare
results with the previous release and investigate any 2x regression even when
the absolute budget passes.

Load a synthetic 100,000-trial checkpoint in the dashboard. Initial render must
finish within 5 seconds, controls must respond within 200 ms after rendering,
and an SSE burst must not regress to an older snapshot. Record the browser's
peak heap and compare it with the previous release.

## Operations and recovery

- Start a token-protected server on a non-loopback interface. Verify a
  disallowed `Origin` receives 403 for both reads and mutations and does not
  change pending/completed counts.
- Run a worker, kill it mid-trial, let its lease expire, and verify the work is
  reclaimed. Restart an exec worker with a nonempty outbox and verify the exact
  tell is replayed without a second completion event.
- Save and restore a checkpoint containing pending work, idempotency keys,
  completion receipts, and non-finite observations. Verify IDs and retry
  behavior survive restart.
- Keep an SSE client connected, send SIGTERM, and verify the server closes the
  stream and exits within 10 seconds. For one forced checkpoint error, verify
  the structured error log carries the response `x-request-id` and the
  aggregate `hola_checkpoint_failures_total` metric increments. Metrics are
  intentionally low-cardinality and do not carry per-request IDs.

## Artifacts

- Inspect the Cargo crates, wheel, and source distribution for README,
  Apache-2.0 license, repository links, and version consistency.
- Record clean-build wall time and compressed artifact sizes beside the prior
  release. The automated package check caps wheels at 64 MiB and source
  distributions at 5 MiB; investigate any 25% size increase or 2x clean-build
  slowdown even when those absolute limits pass.
- Install artifacts in clean environments without the repository on
  `PYTHONPATH`; run the public Python tests and a CLI ask/tell smoke test.
- Confirm the dashboard files bundled in the wheel exactly match the reviewed
  standalone assets.
- Verify GitHub build provenance for every wheel, source distribution, and CLI
  archive, binding it to this workflow and release commit:

  ```bash
  gh attestation verify PATH_TO_ASSET \
    --repo blackrock/HOLA \
    --signer-workflow blackrock/HOLA/.github/workflows/release.yml \
    --source-digest RELEASE_SHA
  ```
