// Copyright 2026 BlackRock, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ============================================================================
// State
// ============================================================================
const S = {
    mode: 'disconnected',   // 'disconnected' | 'live' | 'offline'
    serverUrl: '',
    sse: null,                // AbortController for the live event stream

    trials: [],              // CompletedTrial[] from /api/trials
    space: [],               // [{name, type, min, max, scale}]
    objectives: [],           // [{field, type, target, limit, priority, group}]
    serverObjectives: [],    // snapshot of objectives as fetched from the server
    paramNames: [],
    metricNames: [],
    bestIdx: -1,
    uplot: null,
    sortCol: null,
    sortAsc: true,
    lastTrialTime: null,
    previewActive: false,    // true when client-side rescalarization is active
};

// ============================================================================
// Connection
// ============================================================================
// API token is kept in memory only (never localStorage) so it cannot be read
// by extensions or persisted XSS. If supplied via ?token= for convenience it is
// captured once and stripped from the URL to avoid leaking through history,
// referrer headers, and access logs.
let _apiToken = '';

function captureUrlToken() {
    const params = new URLSearchParams(window.location.search);
    const urlToken = params.get('token');
    if (!urlToken) return;
    _apiToken = urlToken;
    params.delete('token');
    const query = params.toString();
    const newUrl = window.location.pathname + (query ? '?' + query : '') + window.location.hash;
    window.history.replaceState(null, '', newUrl);
}

function apiToken() {
    return _apiToken;
}

function setApiToken(token) {
    _apiToken = token || '';
}

function clearApiToken() {
    _apiToken = '';
    const field = document.getElementById('api-token');
    if (field) field.value = '';
}

function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    const token = apiToken();
    if (token) headers.set('Authorization', `Bearer ${token}`);
    return fetch(url, { ...options, headers });
}

function clearElement(el) {
    el.replaceChildren();
}

// Single-pass min/max. Math.min/max with the spread operator throws RangeError
// (call stack exceeded) once the array is large enough (~100k+ elements), so we
// fold instead of spreading. Returns {min, max} (both NaN for an empty array).
function minMax(arr) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v < min) min = v;
        if (v > max) max = v;
    }
    if (min === Infinity) return { min: NaN, max: NaN };
    return { min, max };
}

async function connectToServer() {
    const url = document.getElementById('server-url').value.trim().replace(/\/+$/, '')
        || 'http://localhost:8000';
    document.getElementById('server-url').value = url;
    S.serverUrl = url;

    const tokenField = document.getElementById('api-token');
    if (tokenField && tokenField.value) setApiToken(tokenField.value);

    try {
        // Probe server by fetching space. All reads go through apiFetch so the
        // bearer token rides along when the server opts into read auth.
        const spaceResp = await apiFetch(`${url}/api/space`);
        if (spaceResp.status === 401) throw new Error('Authentication failed: missing or invalid API token');
        if (!spaceResp.ok) throw new Error('Server not responding');
        const spaceData = await spaceResp.json();
        S.space = spaceData.params || [];
        S.paramNames = S.space.map(p => p.name);

        // Fetch objectives
        const objResp = await apiFetch(`${url}/api/objectives`);
        const objData = await objResp.json();
        S.objectives = objData.objectives || [];
        S.serverObjectives = JSON.parse(JSON.stringify(S.objectives));

        // Fetch trials
        const trialsResp = await apiFetch(`${url}/api/trials?sorted_by=index&include_infeasible=true`);
        S.trials = await trialsResp.json();

        discoverMetrics();
        setMode('live');
        renderAll();
        startStream();
    } catch (e) {
        alert('Failed to connect: ' + e.message);
    }
}

// Live event stream. EventSource cannot send an Authorization header, so we
// stream /api/events via fetch (which carries the bearer token through
// apiFetch) and parse the text/event-stream body incrementally. This keeps the
// token out of the URL and works whether or not read auth is enabled. The
// stream reconnects on drop.
function startStream() {
    stopStream();
    const controller = new AbortController();
    S.sse = controller;
    streamEvents(controller);
}

function stopStream() {
    if (S.sse) {
        S.sse.abort();
        S.sse = null;
    }
}

async function streamEvents(controller) {
    try {
        const resp = await apiFetch(`${S.serverUrl}/api/events`, {
            headers: { Accept: 'text/event-stream' },
            signal: controller.signal,
        });
        if (!resp.ok || !resp.body) {
            setDot('disconnected');
            scheduleReconnect(controller);
            return;
        }
        setDot('connected');
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // Events are separated by a blank line; process each complete one
            // and keep any partial trailing event in the buffer.
            let sep;
            while ((sep = buffer.indexOf('\n\n')) >= 0) {
                const chunk = buffer.slice(0, sep);
                buffer = buffer.slice(sep + 2);
                handleStreamEvent(chunk);
            }
        }
        // Stream ended cleanly; reconnect unless this stream was superseded.
        setDot('disconnected');
        scheduleReconnect(controller);
    } catch (e) {
        if (controller.signal.aborted) return; // intentionally stopped
        setDot('disconnected');
        scheduleReconnect(controller);
    }
}

function scheduleReconnect(controller) {
    // Only reconnect if this controller is still the active stream.
    if (S.sse !== controller || S.mode !== 'live') return;
    setTimeout(() => {
        if (S.sse === controller && S.mode === 'live') streamEvents(controller);
    }, 2000);
}

// Parse one text/event-stream record (lines separated by \n) and dispatch its
// JSON data payload to handleEngineEvent.
function handleStreamEvent(chunk) {
    const dataLines = [];
    for (const line of chunk.split('\n')) {
        if (line.startsWith('data:')) {
            // Strip the "data:" prefix and a single optional leading space.
            dataLines.push(line.slice(line[5] === ' ' ? 6 : 5));
        }
        // Lines starting with ':' are keep-alive comments; ignore them.
    }
    if (dataLines.length === 0) return;
    let event;
    try {
        event = JSON.parse(dataLines.join('\n'));
    } catch {
        return;
    }
    handleEngineEvent(event);
}

async function handleEngineEvent(event) {
    if (event.type === 'TrialCompleted') {
        const trial = event.trial || await fetchCompletedTrial(event.trial_id);
        if (!trial) return;
        upsertTrial(trial);
        discoverMetrics();
        S.lastTrialTime = Date.now();
        if (S.previewActive) previewObjectives(); else renderAll();
    }
}

async function fetchCompletedTrial(trialId) {
    const resp = await apiFetch(`${S.serverUrl}/api/trial/${trialId}?include_infeasible=true`);
    if (!resp.ok) return null;
    return resp.json();
}

function upsertTrial(trial) {
    const idx = S.trials.findIndex(t => t.trial_id === trial.trial_id);
    if (idx >= 0) S.trials[idx] = trial;
    else S.trials.push(trial);
}

function loadCheckpointFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    // Guard against oversized files: readAsText buffers the whole file into a
    // string on the main thread, so a multi-hundred-MB drop would hang the tab
    // or exhaust memory. Reject anything beyond a reasonable checkpoint size.
    const MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024; // 8 MB
    if (file.size > MAX_CHECKPOINT_BYTES) {
        alert(`Checkpoint file is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). ` +
            `Maximum supported size is ${MAX_CHECKPOINT_BYTES / (1024 * 1024)} MB.`);
        event.target.value = '';
        return;
    }
    // Tear down any live stream so its events do not bleed into offline data.
    stopStream();
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            // Support both Checkpoint and LeaderboardCheckpoint formats
            const lb = data.leaderboard;
            if (!lb || !lb.trials) throw new Error('Invalid checkpoint format');
            // Map persisted/checkpoint formats to the CompletedTrial shape.
            // A real persisted leaderboard carries only `observation` (the
            // engine's per-group scores: a scalar or a {group: score} map) plus
            // `raw_metrics`; it has no rank/pareto_front/score_vector. Derive
            // score_vector from the observation so single-objective checkpoints
            // are detected, then compute rank/pareto_front client-side below.
            S.trials = lb.trials.map((t, i) => ({
                trial_id: t.trial_id ?? t.id ?? i,
                params: t.candidate ?? t.params ?? {},
                metrics: t.raw_metrics ?? t.metrics ?? {},
                scores: t.scores ?? {},
                score_vector: t.score_vector ?? observationToScoreVector(t.observation),
                rank: t.rank,
                pareto_front: t.pareto_front,
                completed_at: t.timestamp ?? t.completed_at ?? 0,
            }));
            computeRanksIfMissing(S.trials);
            // Try to extract param names from first trial
            S.space = [];
            S.paramNames = [];
            if (S.trials.length > 0 && S.trials[0].params) {
                const c = typeof S.trials[0].params === 'object' ? S.trials[0].params : {};
                S.paramNames = Object.keys(c);
                S.space = S.paramNames.map(name => ({
                    name, type: 'real', min: 0, max: 1, scale: 'linear'
                }));
                // Compute actual bounds from data
                for (const p of S.space) {
                    const vals = S.trials.map(t => t.params?.[p.name]).filter(v => v != null);
                    if (vals.length > 0) {
                        const mm = minMax(vals);
                        p.min = mm.min;
                        p.max = mm.max;
                    }
                }
            }
            S.objectives = [];
            discoverMetrics();
            setMode('offline');
            renderAll();
        } catch (err) {
            alert('Failed to parse checkpoint: ' + err.message);
        }
    };
    reader.readAsText(file);
    event.target.value = ''; // Reset so same file can be loaded again
}

// Convert a persisted leaderboard observation into the dashboard's score_vector
// shape ({group: score}). The engine stores either a scalar (single objective)
// or a {group: score} map (multi-group). Anything else yields an empty vector.
function observationToScoreVector(observation) {
    if (typeof observation === 'number') return { score: observation };
    if (observation === 'inf') return { score: Infinity };
    if (observation && typeof observation === 'object') return { ...observation };
    return {};
}

// Compute rank and pareto_front client-side for any trial that lacks them (a
// real persisted leaderboard has neither). Defaulting rank to the insertion
// index would always flag trial 0 as best, so we rank off the derived scores
// instead. Trials that already carry server-provided rank are left untouched.
function computeRanksIfMissing(trials) {
    const needs = trials.filter(t => typeof t.rank !== 'number');
    if (needs.length === 0) return;

    const groupsOf = t => {
        const sv = t.score_vector;
        return sv && typeof sv === 'object' ? Object.keys(sv) : [];
    };
    const groupNames = new Set();
    for (const t of trials) for (const g of groupsOf(t)) groupNames.add(g);
    const scoreVal = (t, g) => {
        const v = t.score_vector?.[g];
        if (v === 'inf') return Infinity;
        return typeof v === 'number' ? v : NaN;
    };

    if (groupNames.size <= 1) {
        // Scalar case: order by the sole score ascending (lower is better).
        const g = [...groupNames][0];
        const sumScore = t => {
            if (g != null) return scoreVal(t, g);
            // No groups at all: sum whatever numeric scores exist.
            return Object.values(t.score_vector || {}).reduce(
                (acc, v) => acc + (typeof v === 'number' ? v : 0), 0);
        };
        const order = trials
            .map((t, i) => ({ t, i, s: sumScore(t) }))
            .sort((a, b) => {
                const av = isFinite(a.s) ? a.s : Infinity;
                const bv = isFinite(b.s) ? b.s : Infinity;
                return av - bv || a.i - b.i;
            });
        order.forEach((entry, rank) => {
            if (typeof entry.t.rank !== 'number') entry.t.rank = rank;
            if (typeof entry.t.pareto_front !== 'number') entry.t.pareto_front = rank;
        });
        return;
    }

    // Multi-group case: client-side non-dominated rank. A trial is rank 0 when
    // no other trial dominates it (every group <= and at least one <). Higher
    // ranks count how many trials dominate it, so the front is pareto_front 0.
    const cols = [...groupNames];
    const dominates = (a, b) => {
        let strictly = false;
        for (const g of cols) {
            const av = scoreVal(a, g);
            const bv = scoreVal(b, g);
            if (!isFinite(av) || !isFinite(bv)) return false;
            if (av > bv) return false;
            if (av < bv) strictly = true;
        }
        return strictly;
    };
    for (const t of trials) {
        if (typeof t.rank === 'number') continue;
        let dominatedBy = 0;
        for (const other of trials) {
            if (other === t) continue;
            if (dominates(other, t)) dominatedBy++;
        }
        t.rank = dominatedBy;
        t.pareto_front = dominatedBy === 0 ? 0 : dominatedBy;
    }
}

function discoverMetrics() {
    const names = new Set();
    for (const t of S.trials) {
        if (t.metrics && typeof t.metrics === 'object') {
            for (const k of Object.keys(t.metrics)) names.add(k);
        }
    }
    S.metricNames = [...names].sort();
}

// ============================================================================
// Helpers for score extraction
// ============================================================================

/// True when every trial carries a single comparable scalar score, i.e. there
/// is a single objective group. Summing across multiple groups would collapse
/// incomparable Pareto axes into a meaningless scalar, so callers must treat the
/// multi-objective case via rank / pareto_front instead.
function isSingleObjective() {
    let count = 0;
    for (const t of S.trials) {
        const sv = t.score_vector;
        if (sv && typeof sv === 'object') {
            count = Math.max(count, Object.keys(sv).length);
            if (count > 1) return false;
        }
    }
    return count === 1;
}

/// Get the single scalar score from a trial's score_vector. Returns a finite
/// number only when there is exactly one objective group; for multi-objective
/// trials it returns NaN because no scalar is meaningful across Pareto axes.
function getTrialScore(trial) {
    const sv = trial.score_vector;
    if (!sv || typeof sv !== 'object') return NaN;
    const vals = Object.values(sv).map(v =>
        typeof v === 'number' ? v : v === 'inf' ? Infinity : NaN
    );
    if (vals.length !== 1) return NaN;
    return vals[0];
}

/// The authoritative "best" trial. The server provides rank and pareto_front;
/// the rank-0 / pareto_front-0 trial is the best (or a Pareto-optimal trial in
/// the multi-objective case) without inventing a cross-axis scalar.
function getTrialRank(trial) {
    const r = typeof trial.rank === 'number' ? trial.rank : NaN;
    return isFinite(r) ? r : Infinity;
}

// ============================================================================
// UI State
// ============================================================================
function setMode(mode) {
    S.mode = mode;
    document.getElementById('empty-state').style.display = mode === 'disconnected' ? '' : 'none';
    document.getElementById('main-content').style.display = mode === 'disconnected' ? 'none' : '';
    document.getElementById('mode-label').textContent =
        mode === 'live' ? 'Live' : mode === 'offline' ? 'Offline' : 'Disconnected';
    setDot(mode === 'live' ? 'connected' : mode === 'offline' ? 'offline' : 'disconnected');
    document.getElementById('btn-apply-obj').disabled = mode !== 'live';
    document.getElementById('btn-save-ckpt').disabled = mode !== 'live';
}

function setDot(state) {
    const dot = document.getElementById('sse-dot');
    dot.className = 'dot ' + state;
}

// ============================================================================
// Render All
// ============================================================================
function renderAll() {
    updateStats();
    // Defer chart rendering to the next frame so the layout has been computed
    // (main-content may have just transitioned from display:none to visible).
    requestAnimationFrame(() => {
        renderConvergence();
        renderParetoDropdowns();
        renderPareto();
        renderParallel();
    });
    renderTable();
    renderObjectives();
}

function updateStats() {
    const best = findBest();
    document.getElementById('stat-trials').textContent = S.trials.length;
    // For a single objective the scalar score is meaningful; for multiple
    // objectives report the best trial id (rank 0) instead of a fabricated sum.
    let bestText = '—';
    if (best != null) {
        if (isSingleObjective()) {
            const bestScore = getTrialScore(best);
            if (isFinite(bestScore)) bestText = bestScore.toPrecision(6);
        } else {
            bestText = `#${fmtCell(best.trial_id)}`;
        }
    }
    document.getElementById('stat-best').textContent = bestText;
    if (S.lastTrialTime) {
        const ago = Math.round((Date.now() - S.lastTrialTime) / 1000);
        document.getElementById('stat-last-time').textContent = ago < 60 ? `${ago}s ago` : `${Math.round(ago / 60)}m ago`;
    }
    document.getElementById('table-count').textContent = `${S.trials.length} trials`;
}

// Best trial is driven off the server-provided rank (rank 0 / pareto_front 0)
// rather than any client-side scalarization, which is invalid across multiple
// objectives. Ties (multiple rank-0 Pareto trials) resolve to the first.
function findBest() {
    let best = null;
    let bestIdx = -1;
    let bestRank = Infinity;
    for (let i = 0; i < S.trials.length; i++) {
        const rank = getTrialRank(S.trials[i]);
        if (rank < bestRank) {
            bestRank = rank;
            best = S.trials[i];
            bestIdx = i;
        }
    }
    S.bestIdx = bestIdx;
    return best;
}

// ============================================================================
// Convergence Chart (uPlot)
// ============================================================================
// Re-render charts on window resize
let _resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
        if (S.trials.length > 0) {
            renderConvergence();
            renderPareto();
            renderParallel();
        }
    }, 150);
});

function renderConvergence() {
    const container = document.getElementById('convergence-chart');
    // Measure the card, not the chart container, to avoid uPlot shrinking the
    // container on re-render and creating a feedback loop.
    const card = container.closest('.card');
    const cardW = card ? card.clientWidth - 34 : 0;
    const containerW = container.clientWidth;
    const w = Math.max(cardW, containerW, 400);
    const h = 280;

    // Clear any previous uPlot DOM content so it doesn't accumulate
    clearElement(container);

    const xs = S.trials.map((_, i) => i);
    const single = isSingleObjective();

    // Single objective: plot each trial's scalar score plus the running best.
    // Multi-objective: a single scalar is not comparable across Pareto axes, so
    // instead track the size of the rank-0 Pareto front as trials accumulate,
    // which is a meaningful convergence signal.
    let ys, runBest, scoreLabel, bestLabel;
    if (single) {
        // Convert NaN to null so uPlot treats them as gaps instead of broken values
        ys = S.trials.map(t => {
            const v = getTrialScore(t);
            return isFinite(v) ? v : null;
        });
        runBest = [];
        let best = Infinity;
        for (const y of ys) {
            if (y !== null && y < best) best = y;
            runBest.push(best === Infinity ? null : best);
        }
        scoreLabel = 'Score';
        bestLabel = 'Best';
    } else {
        ys = null;
        runBest = [];
        let frontCount = 0;
        for (const t of S.trials) {
            if (t.pareto_front === 0) frontCount++;
            runBest.push(frontCount);
        }
        bestLabel = 'Pareto front size';
    }

    // Compute y-axis range from finite values only.
    // If the data spans many orders of magnitude, use log scale.
    const plotted = (single ? ys : runBest);
    const finiteAll = plotted.filter(v => v !== null && isFinite(v));
    // A log axis cannot represent 0 (running-best can legitimately reach 0), so
    // if any plotted finite value is exactly 0, force a linear axis.
    const hasZero = finiteAll.some(v => v === 0);
    const finiteYs = finiteAll.filter(v => v > 0);
    let useLog = false;
    let yScaleRange;
    if (finiteYs.length > 1) {
        const { min: posMin, max: yMax } = minMax(finiteYs);
        // Range still spans every finite value (including 0) so the axis frames
        // the actual data, while the log-scale decision uses only positives.
        const { min: yMin } = minMax(finiteAll);
        useLog = !hasZero && yMax / (posMin || 1) > 1000;
        if (!useLog) {
            const yPad = (yMax - yMin) * 0.05 || 1;
            yScaleRange = (u, dataMin, dataMax) => [yMin - yPad, yMax + yPad];
        }
    }

    if (S.uplot) { S.uplot.destroy(); S.uplot = null; }

    const opts = {
        width: w,
        height: h,
        cursor: { show: true, drag: { x: true, y: false } },
        scales: {
            x: { time: false },
            y: useLog
                ? { distr: 3, log: 10 }
                : yScaleRange ? { range: yScaleRange } : {},
        },
        axes: [
            {
                stroke: '#7070a0', grid: { stroke: 'rgba(255,255,255,0.04)' },
                font: '11px Inter', ticks: { stroke: 'rgba(255,255,255,0.06)' }
            },
            {
                stroke: '#7070a0', grid: { stroke: 'rgba(255,255,255,0.04)' },
                font: '11px Inter', ticks: { stroke: 'rgba(255,255,255,0.06)' },
                size: 60,
                values: (u, vals) => vals.map(v => {
                    if (v == null) return '';
                    if (v === 0) return '0';
                    if (Math.abs(v) >= 1000 || Math.abs(v) < 0.01) return v.toExponential(1);
                    return v.toPrecision(3);
                }),
            },
        ],
        series: single
            ? [
                { label: 'Trial' },
                {
                    label: scoreLabel, stroke: '#6c5ce7', width: 1.5,
                    points: { show: true, size: 4, fill: '#6c5ce7' }
                },
                {
                    label: bestLabel, stroke: '#00cec9', width: 2, dash: [6, 3],
                    points: { show: false }
                },
            ]
            : [
                { label: 'Trial' },
                {
                    label: bestLabel, stroke: '#00cec9', width: 2,
                    points: { show: false }
                },
            ],
    };

    const data = single ? [xs, ys, runBest] : [xs, runBest];
    S.uplot = new uPlot(opts, data, container);
}

// ============================================================================
// Pareto Scatter (canvas)
// ============================================================================
function renderParetoDropdowns() {
    const xSel = document.getElementById('pareto-x');
    const ySel = document.getElementById('pareto-y');
    const allFields = [...S.metricNames];
    const prevX = xSel.value;
    const prevY = ySel.value;
    clearElement(xSel);
    clearElement(ySel);
    for (const f of allFields) {
        xSel.add(new Option(f, f, false, f === prevX));
        ySel.add(new Option(f, f, false, f === prevY));
    }
    if (!prevX && allFields.length >= 2) {
        xSel.value = allFields[0];
        ySel.value = allFields.length > 1 ? allFields[1] : allFields[0];
    }
    xSel.onchange = ySel.onchange = renderPareto;
}

function renderPareto() {
    const canvas = document.getElementById('pareto-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth || 500;
    const h = 280;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const xField = document.getElementById('pareto-x').value;
    const yField = document.getElementById('pareto-y').value;
    if (!xField || !yField || S.trials.length === 0) return;

    const pad = { top: 20, right: 20, bottom: 35, left: 55 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    // Extract values from metrics
    const points = S.trials.map((t, i) => ({
        x: getMetric(t, xField),
        y: getMetric(t, yField),
        idx: i,
        onFront: t.pareto_front === 0,
    })).filter(p => isFinite(p.x) && isFinite(p.y));

    if (points.length === 0) return;

    // Scale
    const xmm = minMax(points.map(p => p.x));
    const ymm = minMax(points.map(p => p.y));
    let xMin = xmm.min, xMax = xmm.max;
    let yMin = ymm.min, yMax = ymm.max;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    xMin -= xRange * 0.05; xMax += xRange * 0.05;
    yMin -= yRange * 0.05; yMax += yRange * 0.05;

    const sx = v => pad.left + (v - xMin) / (xMax - xMin) * pw;
    const sy = v => pad.top + ph - (v - yMin) / (yMax - yMin) * ph;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const gx = pad.left + (pw / 4) * i;
        const gy = pad.top + (ph / 4) * i;
        ctx.beginPath(); ctx.moveTo(gx, pad.top); ctx.lineTo(gx, pad.top + ph); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + pw, gy); ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = '#7070a0';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(xField, pad.left + pw / 2, h - 5);
    ctx.save();
    ctx.translate(12, pad.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yField, 0, 0);
    ctx.restore();

    // Axis tick labels
    ctx.fillStyle = '#555';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
        const xVal = xMin + (xMax - xMin) * (i / 4);
        const gx = pad.left + (pw / 4) * i;
        ctx.fillText(fmt(xVal), gx, pad.top + ph + 14);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const yVal = yMin + (yMax - yMin) * (i / 4);
        const gy = pad.top + ph - (ph / 4) * i;
        ctx.fillText(fmt(yVal), pad.left - 6, gy + 3);
    }

    // Draw the Pareto front connecting line only for a true 2-objective view,
    // i.e. when the optimization has exactly two objective fields and the chosen
    // axes are those fields. Sorting front points on an arbitrary axis pair would
    // produce a misleading zig-zag, so for any other axis pair we only mark the
    // pareto_front == 0 points (drawn below) without connecting them.
    const front = points.filter(p => p.onFront);
    if (front.length >= 2 && isParetoAxisPair(xField, yField)) {
        const sorted = [...front].sort((a, b) => a.x - b.x);
        ctx.strokeStyle = 'rgba(108, 92, 231, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(sx(sorted[0].x), sy(sorted[0].y));
        for (let i = 1; i < sorted.length; i++) {
            ctx.lineTo(sx(sorted[i].x), sy(sorted[i].y));
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw points and store positions for tooltip
    S._paretoPoints = [];
    for (const p of points) {
        const px = sx(p.x);
        const py = sy(p.y);
        ctx.beginPath();
        ctx.arc(px, py, p.onFront ? 5 : 3, 0, Math.PI * 2);
        ctx.fillStyle = p.onFront ? '#6c5ce7' : '#444466';
        ctx.fill();
        if (p.onFront) {
            ctx.strokeStyle = '#a29bfe';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        S._paretoPoints.push({
            cx: px, cy: py,
            trial: S.trials[p.idx],
            xVal: p.x, yVal: p.y,
            xField, yField,
            onFront: p.onFront,
        });
    }

    // Attach tooltip listener (once)
    attachParetoTooltip();
}

function getMetric(trial, field) {
    const v = trial.metrics?.[field];
    if (v === 'inf') return Infinity;
    return v ?? NaN;
}

// The Pareto front is a true frontier in objective space only. Connecting the
// front points is meaningful only when there are exactly two objectives and the
// selected scatter axes are precisely those two objective fields.
function isParetoAxisPair(xField, yField) {
    const fields = S.objectives
        .map(o => o.field)
        .filter(f => f != null);
    if (fields.length !== 2) return false;
    const set = new Set(fields);
    return xField !== yField && set.has(xField) && set.has(yField);
}

// ============================================================================
// Pareto Tooltip
// ============================================================================
let _paretoTooltipAttached = false;
function attachParetoTooltip() {
    if (_paretoTooltipAttached) return;
    const canvas = document.getElementById('pareto-canvas');
    const tooltip = document.getElementById('pareto-tooltip');
    if (!canvas || !tooltip) return;
    _paretoTooltipAttached = true;

    canvas.addEventListener('mousemove', (e) => {
        if (!S._paretoPoints || S._paretoPoints.length === 0) {
            tooltip.classList.remove('visible');
            return;
        }
        const rect = canvas.getBoundingClientRect();
        // Mouse position in CSS pixels relative to canvas
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Find nearest point within 10px threshold
        let nearest = null;
        let nearestDist = Infinity;
        for (const p of S._paretoPoints) {
            const dx = p.cx - mx;
            const dy = p.cy - my;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < nearestDist) {
                nearestDist = dist;
                nearest = p;
            }
        }

        if (nearest && nearestDist <= 10) {
            clearElement(tooltip);

            const title = document.createElement('strong');
            title.textContent = `Trial ${fmtCell(nearest.trial.trial_id)}`;

            const xLine = document.createElement('div');
            xLine.textContent = `${nearest.xField}: ${fmtCell(nearest.xVal)}`;

            const yLine = document.createElement('div');
            yLine.textContent = `${nearest.yField}: ${fmtCell(nearest.yVal)}`;

            const status = document.createElement('span');
            status.textContent = nearest.onFront ? 'Pareto front' : 'Dominated';
            status.style.color = nearest.onFront ? 'var(--accent-bright)' : 'var(--text-2)';

            tooltip.append(title, xLine, yLine, status);
            // Position tooltip near cursor but keep it inside the container
            const container = canvas.parentElement;
            const cw = container.clientWidth;
            let tx = mx + 12;
            let ty = my - 10;
            // Prevent overflow on right side
            if (tx + 160 > cw) tx = mx - 170;
            if (ty < 0) ty = 0;
            tooltip.style.left = tx + 'px';
            tooltip.style.top = ty + 'px';
            tooltip.classList.add('visible');
        } else {
            tooltip.classList.remove('visible');
        }
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.classList.remove('visible');
    });
}

// ============================================================================
// Parallel Coordinates (canvas)
// ============================================================================
function renderParallel() {
    const card = document.getElementById('parallel-card');
    if (S.paramNames.length < 2) {
        card.style.display = 'none';
        return;
    }
    card.style.display = '';

    const canvas = document.getElementById('parallel-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth - 32;
    const h = 200;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const pad = { top: 25, bottom: 30, left: 40, right: 40 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;
    const names = S.paramNames;
    const n = names.length;
    const gap = pw / (n - 1);

    // Compute min/max per param, with categorical support
    const ranges = names.map(name => {
        const sp = S.space.find(s => s.name === name);
        if (sp && sp.type === 'categorical' && sp.choices) {
            return { min: 0, max: sp.choices.length - 1, categorical: true, choices: sp.choices };
        }
        if (sp) return { min: sp.min, max: sp.max };
        const vals = S.trials.map(t => t.params?.[name]).filter(v => v != null);
        return minMax(vals);
    });

    // Coloring metric: the scalar score when single-objective, otherwise the
    // server-provided rank (lower is better) so multi-objective lines still
    // convey relative quality without an invalid cross-axis scalar.
    const single = isSingleObjective();
    const colorValue = single ? (t => getTrialScore(t)) : (t => getTrialRank(t));
    const scores = S.trials.map(colorValue).filter(isFinite);
    const { min: sMin, max: sMax } = minMax(scores);
    const sRange = sMax - sMin || 1;

    // Draw axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.font = '10px Inter';
    ctx.fillStyle = '#7070a0';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) {
        const x = pad.left + i * gap;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + ph);
        ctx.stroke();
        ctx.fillText(names[i], x, h - 8);
        // Min/max labels (show choice labels for categorical)
        ctx.fillStyle = '#555';
        if (ranges[i].categorical && ranges[i].choices) {
            const choices = ranges[i].choices;
            ctx.fillText(choices[choices.length - 1], x, pad.top - 6);
            ctx.fillText(choices[0], x, pad.top + ph + 14);
        } else {
            ctx.fillText(fmt(ranges[i].max), x, pad.top - 6);
            ctx.fillText(fmt(ranges[i].min), x, pad.top + ph + 14);
        }
        ctx.fillStyle = '#7070a0';
    }

    // Resolve a param value to a numeric value for the axis
    function resolveParamValue(rawVal, rangeInfo) {
        if (rangeInfo.categorical && rangeInfo.choices) {
            const idx = rangeInfo.choices.indexOf(rawVal);
            return idx >= 0 ? idx : 0;
        }
        return typeof rawVal === 'number' ? rawVal : 0;
    }

    // Draw lines
    for (const trial of S.trials) {
        const score = colorValue(trial);
        if (!isFinite(score)) continue;
        const t = (score - sMin) / sRange;
        const r = Math.round(108 + t * (68 - 108));
        const g = Math.round(92 + t * (68 - 92));
        const b = Math.round(231 + t * (102 - 231));
        ctx.strokeStyle = `rgba(${r},${g},${b},0.35)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const x = pad.left + i * gap;
            const val = resolveParamValue(trial.params?.[names[i]] ?? 0, ranges[i]);
            const { min, max } = ranges[i];
            const range = max - min || 1;
            const y = pad.top + ph - ((val - min) / range) * ph;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Highlight best
    if (S.bestIdx >= 0) {
        const trial = S.trials[S.bestIdx];
        ctx.strokeStyle = '#00cec9';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const x = pad.left + i * gap;
            const val = resolveParamValue(trial.params?.[names[i]] ?? 0, ranges[i]);
            const { min, max } = ranges[i];
            const range = max - min || 1;
            const y = pad.top + ph - ((val - min) / range) * ph;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}

function fmt(v) {
    if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(1);
    return Number(v.toPrecision(4)).toString();
}

// ============================================================================
// Trial Table
// ============================================================================
function renderTable() {
    const thead = document.getElementById('trial-thead').querySelector('tr');
    const tbody = document.getElementById('trial-tbody');

    // Build columns
    const cols = ['trial_id', 'rank', ...S.paramNames, ...S.metricNames];

    clearElement(thead);
    for (const c of cols) {
        const th = document.createElement('th');
        th.textContent = c;
        if (S.sortCol === c) th.className = S.sortAsc ? 'sorted-asc' : 'sorted-desc';
        th.addEventListener('click', () => sortTable(c));
        thead.appendChild(th);
    }

    // Sort trials
    let sorted = [...S.trials];
    if (S.sortCol) {
        sorted.sort((a, b) => compareCellValues(
            getCellValue(a, S.sortCol),
            getCellValue(b, S.sortCol),
            S.sortAsc,
        ));
    }

    clearElement(tbody);
    for (const t of sorted) {
        const isBest = t.trial_id === (S.bestIdx >= 0 ? S.trials[S.bestIdx].trial_id : -1);
        const tr = document.createElement('tr');
        if (isBest) tr.className = 'best-row';
        for (const c of cols) {
            const td = document.createElement('td');
            td.textContent = fmtCell(getCellValue(t, c));
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    }
}

function getCellValue(trial, col) {
    if (col === 'trial_id') return trial.trial_id;
    if (col === 'rank') return trial.rank;
    if (S.paramNames.includes(col)) return trial.params?.[col] ?? NaN;
    return trial.metrics?.[col] ?? NaN;
}

// Normalize a raw cell value to a sortable form: the string 'inf'/'-inf'
// sentinels become +/-Infinity (matching getMetric), everything else is left
// as-is. Numbers (finite or Infinity) sort numerically; non-numeric/missing
// values are handled separately by compareCellValues.
function normalizeSortValue(v) {
    if (v === 'inf') return Infinity;
    if (v === '-inf') return -Infinity;
    return v;
}

// Deterministic ordering for table cells. Numeric values (including +/-Infinity
// via the 'inf' sentinels) compare numerically; remaining values compare as
// strings via localeCompare. NaN/null/undefined always sort to the very end of
// the table regardless of `asc`, so the missing-value handling is applied after
// the ascending/descending decision rather than being inverted by it.
function compareCellValues(a, b, asc) {
    const na = normalizeSortValue(a);
    const nb = normalizeSortValue(b);
    const aMissing = na == null || (typeof na === 'number' && isNaN(na));
    const bMissing = nb == null || (typeof nb === 'number' && isNaN(nb));
    if (aMissing || bMissing) {
        if (aMissing && bMissing) return 0;
        return aMissing ? 1 : -1; // missing last in both directions
    }
    let cmp;
    if (typeof na === 'number' && typeof nb === 'number') {
        cmp = na < nb ? -1 : na > nb ? 1 : 0;
    } else {
        cmp = String(na).localeCompare(String(nb));
    }
    return asc ? cmp : -cmp;
}

function fmtCell(v) {
    if (v === 'inf') return '∞';
    if (v == null || (typeof v === 'number' && isNaN(v))) return '—';
    if (typeof v === 'number') {
        if (!isFinite(v)) return '∞';
        return Number(v.toPrecision(6)).toString();
    }
    return String(v);
}

function sortTable(col) {
    if (S.sortCol === col) S.sortAsc = !S.sortAsc;
    else { S.sortCol = col; S.sortAsc = true; }
    renderTable();
}

// ============================================================================
// Objective Controls
// ============================================================================
function renderObjectives() {
    const container = document.getElementById('objectives-list');
    clearElement(container);
    if (S.objectives.length === 0) {
        const empty = document.createElement('div');
        empty.style.color = 'var(--text-2)';
        empty.style.fontSize = '0.82rem';
        empty.style.padding = '8px 0';
        empty.textContent = 'No objectives configured';
        container.appendChild(empty);
        return;
    }
    S.objectives.forEach((obj, i) => {
        const row = document.createElement('div');
        row.className = 'objective-row';

        const field = document.createElement('span');
        field.className = 'obj-field';
        field.textContent = obj.field ?? '';

        const type = document.createElement('span');
        type.className = 'obj-type';
        type.textContent = obj.obj_type || obj.type || 'minimize';

        const priorityLabel = makeObjectiveLabel('Priority');
        const priority = document.createElement('input');
        priority.type = 'range';
        priority.min = '0';
        priority.max = '5';
        priority.step = '0.1';
        priority.value = obj.priority ?? 1;
        const priorityValue = document.createElement('span');
        priorityValue.className = 'obj-priority-value';
        priorityValue.textContent = priority.value;
        priority.addEventListener('input', () => {
            S.objectives[i].priority = parseFloat(priority.value);
            priorityValue.textContent = priority.value;
        });
        priorityLabel.append(priority, priorityValue);

        const targetLabel = makeObjectiveLabel('Target');
        const target = document.createElement('input');
        target.type = 'number';
        target.step = 'any';
        target.value = obj.target ?? '';
        target.addEventListener('change', () => {
            S.objectives[i].target = target.value ? parseFloat(target.value) : null;
        });
        targetLabel.appendChild(target);

        const limitLabel = makeObjectiveLabel('Limit');
        const limit = document.createElement('input');
        limit.type = 'number';
        limit.step = 'any';
        limit.value = obj.limit ?? '';
        limit.addEventListener('change', () => {
            S.objectives[i].limit = limit.value ? parseFloat(limit.value) : null;
        });
        limitLabel.appendChild(limit);

        const groupLabel = makeObjectiveLabel('Group');
        const group = document.createElement('input');
        group.type = 'text';
        group.className = 'obj-group-input';
        group.value = obj.group ?? '';
        group.addEventListener('change', () => {
            S.objectives[i].group = group.value || null;
        });
        groupLabel.appendChild(group);

        row.append(field, type, priorityLabel, targetLabel, limitLabel, groupLabel);
        container.appendChild(row);
    });
}

function makeObjectiveLabel(text) {
    const label = document.createElement('label');
    label.className = 'objective-label';
    label.append(document.createTextNode(text));
    return label;
}

// Client-side TLP rescalarization for preview mode.
async function resetObjectives() {
    S.objectives = JSON.parse(JSON.stringify(S.serverObjectives));
    S.previewActive = false;
    document.getElementById('preview-badge').style.display = 'none';
    if (S.mode === 'live') {
        // Re-fetch trials with the server's actual scores
        const resp = await apiFetch(`${S.serverUrl}/api/trials?sorted_by=index&include_infeasible=true`);
        S.trials = await resp.json();
    }
    renderAll();
}

function previewObjectives() {
    S.previewActive = true;
    document.getElementById('preview-badge').style.display = '';
    for (const trial of S.trials) {
        const m = trial.metrics;
        if (!m || typeof m !== 'object') continue;
        const groups = {};
        let feasible = true;
        for (const obj of S.objectives) {
            // Normalize the engine's string sentinels to match getMetric so a
            // persisted 'inf'/'-inf' metric is treated as +/-Infinity (and thus
            // infeasible) rather than as a non-numeric value.
            let raw = m[obj.field];
            if (raw === 'inf') raw = Infinity;
            else if (raw === '-inf') raw = -Infinity;
            if (raw == null || !isFinite(raw)) { feasible = false; continue; }
            const isMin = (obj.obj_type || obj.type || 'minimize') === 'minimize';
            let score;
            if (obj.target != null && obj.limit != null) {
                const t = obj.target, l = obj.limit;
                const val = isMin ? raw : -raw;
                const tAdj = isMin ? t : -t;
                const lAdj = isMin ? l : -l;
                if (val <= tAdj) score = 0;
                else if (val >= lAdj) { score = Infinity; feasible = false; }
                else score = obj.priority * (val - tAdj) / (lAdj - tAdj);
            } else {
                score = (isMin ? 1 : -1) * obj.priority * raw;
            }
            const g = obj.group || obj.field;
            groups[g] = (groups[g] || 0) + score;
        }
        trial.score_vector = feasible ? groups : Object.fromEntries(
            Object.keys(groups).map(k => [k, null])
        );
    }
    renderAll();
}

async function applyObjectives() {
    if (S.mode !== 'live') return;
    if (!confirm('This will update the server objectives and rescalarize all trials. The server will use these objectives for future sampling. Continue?')) return;
    try {
        const resp = await apiFetch(`${S.serverUrl}/api/objectives`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ objectives: S.objectives }),
        });
        if (!resp.ok) throw new Error('Failed');
        S.previewActive = false;
        document.getElementById('preview-badge').style.display = 'none';
        S.serverObjectives = JSON.parse(JSON.stringify(S.objectives));
        // Re-fetch trials with server-side rescalarization
        const trialsResp = await apiFetch(`${S.serverUrl}/api/trials?sorted_by=index&include_infeasible=true`);
        S.trials = await trialsResp.json();
        renderAll();
    } catch (e) {
        alert('Failed to update objectives: ' + e.message);
    }
}

// ============================================================================
// Checkpoint Controls
// ============================================================================
async function saveCheckpoint() {
    if (S.mode !== 'live') return;
    try {
        const resp = await apiFetch(`${S.serverUrl}/api/checkpoint/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: `Dashboard save at ${new Date().toISOString()}` }),
        });
        const data = await resp.json();
        if (resp.ok) {
            const kind = data.checkpoint_type ? `${data.checkpoint_type} checkpoint` : 'checkpoint';
            alert(`Saved ${kind}: ${data.path} (${data.trials_saved} trials)`);
        }
        else alert('Save failed: ' + (data.error || 'unknown'));
    } catch (e) {
        alert('Save failed: ' + e.message);
    }
}

function exportData() {
    const data = {
        leaderboard: {
            trials: S.trials,
            next_trial_id: S.trials.length,
        },
        metadata: {
            n_trials: S.trials.length,
            created_at_iso: new Date().toISOString(),
            description: 'Exported from HOLA dashboard',
            format_version: 1,
        },
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hola_export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================================================
// Resize handling
// ============================================================================
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        if (S.trials.length > 0) renderAll();
    }, 200);
});

// ============================================================================
// Event wiring and startup
// ============================================================================
// All UI events are bound here via addEventListener so the markup needs no
// inline on* handlers, which lets the page run under a script-src CSP without
// 'unsafe-inline'.
function wireEvents() {
    const openFile = () => document.getElementById('file-input').click();

    document.getElementById('btn-connect').addEventListener('click', connectToServer);
    document.getElementById('btn-clear-token').addEventListener('click', clearApiToken);
    document.getElementById('file-input').addEventListener('change', loadCheckpointFile);

    document.getElementById('btn-preview-obj').addEventListener('click', previewObjectives);
    document.getElementById('btn-reset-obj').addEventListener('click', resetObjectives);
    document.getElementById('btn-apply-obj').addEventListener('click', applyObjectives);
    document.getElementById('btn-save-ckpt').addEventListener('click', saveCheckpoint);
    document.getElementById('btn-export').addEventListener('click', exportData);

    for (const el of document.querySelectorAll('[data-action="connect"]')) {
        el.addEventListener('click', connectToServer);
    }
    for (const el of document.querySelectorAll('[data-action="open-file"]')) {
        el.addEventListener('click', openFile);
    }
}

function startup() {
    wireEvents();
    // Capture a ?token= once into memory and strip it from the URL.
    captureUrlToken();
    // Auto-connect from URL params
    const params = new URLSearchParams(location.search);
    const server = params.get('server');
    if (server) {
        document.getElementById('server-url').value = server;
        connectToServer();
    }
}

startup();

// Update "last trial" timer
setInterval(() => {
    if (S.lastTrialTime) {
        const ago = Math.round((Date.now() - S.lastTrialTime) / 1000);
        document.getElementById('stat-last-time').textContent =
            ago < 60 ? `${ago}s ago` : `${Math.round(ago / 60)}m ago`;
    }
}, 5000);
