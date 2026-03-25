# Dashboard

HOLA includes a browser-based dashboard for real-time monitoring
and offline analysis. It connects to a running server via SSE or
loads checkpoint files directly.

## Opening the dashboard

We serve the dashboard as a static HTML/CSS/JS application. Open
`dashboard/index.html` in any modern browser. No build step or
server is required.

When running a study from Python with `study.serve()`, we can
serve the dashboard automatically by passing the `dashboard_path`
argument. In editable installs from source, point it at the
repository's `dashboard/` directory.

## Connecting to a live server

1. Enter the server URL in the top bar
   (e.g., `http://localhost:8000`)
2. Click **Connect**

The dashboard connects to the server's `/api/events` SSE endpoint
and loads the current state from `/api/trials`, `/api/space`, and
`/api/objectives`. New trials appear in real time as workers report
results.

The status bar shows the following.

**Connection status.**
:   Green dot when connected.

**Trials.**
:   Total completed trials.

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

We plot each trial's score and a running-best curve. The x-axis
is the trial index; the y-axis is the scalarized score. When the
data spans many orders of magnitude, the y-axis switches to a log
scale automatically. Hover over the chart to inspect individual
trials.

### Pareto scatter

We draw a scatter plot of two metrics fields, useful for inspecting
trade-offs in multi-objective optimization. Use the **X** and **Y**
dropdowns to select which metrics to plot. Trials on the first
Pareto front are highlighted and connected with a dashed line.
Hover over a point to see the trial ID and metric values.

### Parallel coordinates

We display all parameter dimensions as parallel vertical axes,
with each trial drawn as a line connecting its parameter values.
Categorical parameters show their choice labels on the axis. Line
color reflects trial quality; better trials are brighter. The
best trial is highlighted in teal.

### Trial table

We provide a sortable table of all completed trials showing trial
ID, rank, parameter values, and metrics. Click any column header
to sort.

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
    mode.

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
