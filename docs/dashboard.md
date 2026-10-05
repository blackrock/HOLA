# Dashboard

HOLA includes a browser-based dashboard for real-time monitoring
and offline analysis. It connects to a running server via SSE or
loads checkpoint files directly.

## Opening the dashboard

The dashboard is a static HTML/CSS/JS application with no build step.
For live monitoring, have HOLA serve the files so the UI and API share
an origin:

```bash
hola serve config.yaml --dashboard ./dashboard
```

Then open `http://localhost:8000/`. To analyze a saved checkpoint
without a live server, you can instead open `dashboard/index.html`
directly in a modern browser.

When running a study from Python with `study.serve()`, we can
serve the dashboard automatically by passing the `dashboard_path`
argument. Wheels and editable installs both include dashboard assets;
use `dashboard_path=str(hola_opt.dashboard_dir())` to select them.

## Connecting to a live server

1. Enter the server URL in the top bar
   (e.g., `http://localhost:8000`)
2. Click **Connect**

If the dashboard is hosted on a different origin, add that exact
origin to the server with `--cors-origin`. Cross-origin browser access
is disabled by default.

The dashboard connects to the server's `/api/events` SSE endpoint
and loads the current state from `/api/trials`, `/api/space`, and
`/api/objectives`. New trials appear in real time as workers report
results.

The status bar shows the following.

**Connection status.**
:   Green dot when connected.

**Displayed trials.**
:   Completed trials retained in the loaded snapshot or received during this
    connection. This is not the study's lifetime completion count when
    leaderboard retention is bounded.

**Best.**
:   Current best score.

**Last.**
:   Time since the most recent trial.

## Loading a checkpoint file

1. Click **Open checkpoint** in the top bar (or the empty-state
   prompt)
2. Select a `.json` checkpoint file

This loads the checkpoint's leaderboard for offline analysis. All
visualizations populate from the stored trials.

## Visualizations

### Convergence plot

For one objective group, we plot each trial's score and a running-best
curve. The x-axis is the trial index; the y-axis is the scalarized score. When the
data spans many orders of magnitude, the y-axis switches to a log
scale automatically. Hover over the chart to inspect individual
trials. For multiple objective groups, the chart shows frontier size
over completion order instead of a scalar running best.

### Pareto scatter

We draw a scatter plot of two metrics fields, useful for inspecting
trade-offs in multi-objective optimization. Use the **X** and **Y**
dropdowns to select which metrics to plot. Trials on the first
Pareto front are highlighted. A dashed connector is shown only for
study with exactly two objective fields and those fields selected as
the raw metric axes. Hover over a point
to see the trial ID and metric values.

Live snapshots retain the server's authoritative ranks. When browser
ranking is needed, frontier membership is exact for one or two objective
groups and for up to 2,048 feasible trials with more groups. Inferred
browser ranks order each frontier by insertion order; server ranks use
NSGA-II crowding distance. Larger unranked
imports and previews use a partial frontier to keep the interface
responsive: highlighted points are verified members; other memberships
and ranks remain unknown. A status message and chart label identify
this case. Exported trials preserve the partial ranking status.

### Parallel coordinates

We display all parameter dimensions as parallel vertical axes,
with each trial drawn as a line connecting its parameter values.
Categorical parameters show their choice labels on the axis. Line
color reflects trial quality; better trials are brighter. The
best trial is highlighted in teal.

### Trial table

We provide a sortable table of completed trials showing trial
ID, rank, parameter values, and metrics. The table shows up to 1,000
rows from the current sort order; charts and exports use the full
loaded population. Click any column header to sort.

## Editing objectives

The **Objectives** panel shows the current objective configuration,
including field, type, priority, target, limit, and group.

We provide three actions.

**Preview.**
:   Rescalarizes all trials in the browser using client-side TLP
    math. The server and its sampling logic are not affected. A
    yellow **PREVIEW** badge appears to indicate that the displayed
    scores differ from the server's. New trials arriving via SSE
    are also rescalarized client-side while preview is active. This
    is useful for exploring "what if" scenarios---for example,
    adjusting priorities or adding constraints---without committing
    the change.

**Reset.**
:   Restores the objectives to the server's current configuration,
    re-fetches trial scores from the server, and exits preview
    mode. In offline mode, it restores the imported objectives,
    scores, ranks, and frontier memberships saved before the preview.

**Apply to server.**
:   Sends the edited objectives to the server via
    `PATCH /api/objectives`. The server rescalarizes all existing
    trials and uses the new objectives for future sampling. A
    confirmation dialog appears before the request is sent. Only
    available in live mode.

## Checkpoints panel

The **Checkpoints** panel provides three actions.

**Open checkpoint.**
:   Load a previously saved `.json` checkpoint file for offline
    analysis.

**Save server state.**
:   Tell the running server to write a full checkpoint (trials,
    strategy state, and configuration) to disk via
    `POST /api/checkpoint/save`. The file is written on the server
    machine.

**Download trials.**
:   Save the trial data currently displayed in the browser as a
    JSON file. This is a browser-side export and does not require
    a server connection.
