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
    sse: null,
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
async function connectToServer() {
    const url = document.getElementById('server-url').value.trim().replace(/\/+$/, '')
        || 'http://localhost:8000';
    document.getElementById('server-url').value = url;
    S.serverUrl = url;

    try {
        // Probe server by fetching space
        const spaceResp = await fetch(`${url}/api/space`);
        if (!spaceResp.ok) throw new Error('Server not responding');
        const spaceData = await spaceResp.json();
        S.space = spaceData.params || [];
        S.paramNames = S.space.map(p => p.name);

        // Fetch objectives
        const objResp = await fetch(`${url}/api/objectives`);
        const objData = await objResp.json();
        S.objectives = objData.objectives || [];
        S.serverObjectives = JSON.parse(JSON.stringify(S.objectives));

        // Fetch trials
        const trialsResp = await fetch(`${url}/api/trials?sorted_by=index&include_infeasible=true`);
        S.trials = await trialsResp.json();

        discoverMetrics();
        setMode('live');
        renderAll();
        startSSE();
    } catch (e) {
        alert('Failed to connect: ' + e.message);
    }
}

function startSSE() {
    if (S.sse) S.sse.close();
    S.sse = new EventSource(`${S.serverUrl}/api/events`);
    S.sse.onopen = () => setDot('connected');
    S.sse.onerror = () => setDot('disconnected');
    S.sse.onmessage = async (e) => {
        const event = JSON.parse(e.data);
        if (event.type === 'TrialCompleted') {
            // Re-fetch trials
            const resp = await fetch(`${S.serverUrl}/api/trials?sorted_by=index&include_infeasible=true`);
            S.trials = await resp.json();
            discoverMetrics();
            S.lastTrialTime = Date.now();
            if (S.previewActive) previewObjectives(); else renderAll();
        }
    };
}

function loadCheckpointFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            // Support both Checkpoint and LeaderboardCheckpoint formats
            const lb = data.leaderboard;
            if (!lb || !lb.trials) throw new Error('Invalid checkpoint format');
            // Map old checkpoint format to new CompletedTrial shape
            S.trials = lb.trials.map((t, i) => ({
                trial_id: t.trial_id ?? t.id ?? i,
                params: t.candidate ?? t.params ?? {},
                metrics: t.raw_metrics ?? t.metrics ?? {},
                scores: t.scores ?? {},
                score_vector: t.score_vector ?? {},
                rank: t.rank ?? i,
                pareto_front: t.pareto_front ?? i,
                completed_at: t.timestamp ?? t.completed_at ?? 0,
            }));
            // Try to extract param names from first trial
            S.space = [];
            S.paramNames = [];
            if (S.trials.length > 0 && S.trials[0].params) {
                const c = typeof S.trials[0].params === 'object' ? S.trials[0].params : {};
                S.paramNames = Object.keys(c);
                S.space = S.paramNames.map(name => ({
                    name, type: 'continuous', min: 0, max: 1, scale: 'linear'
                }));
                // Compute actual bounds from data
                for (const p of S.space) {
                    const vals = S.trials.map(t => t.params?.[p.name]).filter(v => v != null);
                    if (vals.length > 0) {
                        p.min = Math.min(...vals);
                        p.max = Math.max(...vals);
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

/// Get the primary scalar score from a trial's score_vector.
/// For single-objective, returns the only value.
/// For multi-objective, returns the sum of all group scores.
function getTrialScore(trial) {
    const sv = trial.score_vector;
    if (!sv || typeof sv !== 'object') return NaN;
    const vals = Object.values(sv).map(v =>
        typeof v === 'number' ? v : v === 'inf' ? Infinity : NaN
    );
    if (vals.length === 0) return NaN;
    if (vals.length === 1) return vals[0];
    return vals.reduce((a, b) => a + b, 0);
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
    const bestScore = best != null ? getTrialScore(best) : NaN;
    document.getElementById('stat-best').textContent =
        isFinite(bestScore) ? bestScore.toPrecision(6) : '—';
    if (S.lastTrialTime) {
        const ago = Math.round((Date.now() - S.lastTrialTime) / 1000);
        document.getElementById('stat-last-time').textContent = ago < 60 ? `${ago}s ago` : `${Math.round(ago / 60)}m ago`;
    }
    document.getElementById('table-count').textContent = `${S.trials.length} trials`;
}

function findBest() {
    let best = null;
    let bestIdx = -1;
    for (let i = 0; i < S.trials.length; i++) {
        const score = getTrialScore(S.trials[i]);
        if (isFinite(score) && (best === null || score < getTrialScore(best))) {
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
    container.innerHTML = '';

    const xs = S.trials.map((_, i) => i);
    // Convert NaN to null so uPlot treats them as gaps instead of broken values
    const ys = S.trials.map(t => {
        const v = getTrialScore(t);
        return isFinite(v) ? v : null;
    });

    // Running best
    const runBest = [];
    let best = Infinity;
    for (const y of ys) {
        if (y !== null && y < best) best = y;
        runBest.push(best === Infinity ? null : best);
    }

    // Compute y-axis range from finite values only.
    // If the data spans many orders of magnitude, use log scale.
    const finiteYs = ys.filter(v => v !== null && v > 0);
    let useLog = false;
    let yScaleRange;
    if (finiteYs.length > 1) {
        const yMin = Math.min(...finiteYs);
        const yMax = Math.max(...finiteYs);
        useLog = yMax / (yMin || 1) > 1000;
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
        series: [
            { label: 'Trial' },
            {
                label: 'Score', stroke: '#6c5ce7', width: 1.5,
                points: { show: true, size: 4, fill: '#6c5ce7' }
            },
            {
                label: 'Best', stroke: '#00cec9', width: 2, dash: [6, 3],
                points: { show: false }
            },
        ],
    };

    S.uplot = new uPlot(opts, [xs, ys, runBest], container);
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
    xSel.innerHTML = '';
    ySel.innerHTML = '';
    for (const f of allFields) {
        xSel.innerHTML += `<option value="${f}" ${f === prevX ? 'selected' : ''}>${f}</option>`;
        ySel.innerHTML += `<option value="${f}" ${f === prevY ? 'selected' : ''}>${f}</option>`;
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
    let xMin = Math.min(...points.map(p => p.x));
    let xMax = Math.max(...points.map(p => p.x));
    let yMin = Math.min(...points.map(p => p.y));
    let yMax = Math.max(...points.map(p => p.y));
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

    // Draw Pareto front line (using pareto_front == 0 from server)
    const front = points.filter(p => p.onFront);
    if (front.length >= 2) {
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
            tooltip.innerHTML =
                `<strong>Trial ${nearest.trial.trial_id}</strong><br>` +
                `${nearest.xField}: ${fmtCell(nearest.xVal)}<br>` +
                `${nearest.yField}: ${fmtCell(nearest.yVal)}<br>` +
                (nearest.onFront
                    ? '<span style="color:var(--accent-bright)">Pareto front</span>'
                    : '<span style="color:var(--text-2)">Dominated</span>');
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
        return { min: Math.min(...vals), max: Math.max(...vals) };
    });

    // Score range for coloring
    const scores = S.trials.map(t => getTrialScore(t)).filter(isFinite);
    const sMin = Math.min(...scores);
    const sMax = Math.max(...scores);
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
        const score = getTrialScore(trial);
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

    thead.innerHTML = cols.map(c => {
        const cls = S.sortCol === c ? (S.sortAsc ? 'sorted-asc' : 'sorted-desc') : '';
        return `<th class="${cls}" onclick="sortTable('${c}')">${c}</th>`;
    }).join('');

    // Sort trials
    let sorted = [...S.trials];
    if (S.sortCol) {
        sorted.sort((a, b) => {
            const va = getCellValue(a, S.sortCol);
            const vb = getCellValue(b, S.sortCol);
            return S.sortAsc ? va - vb : vb - va;
        });
    }

    tbody.innerHTML = sorted.map(t => {
        const isBest = t.trial_id === (S.bestIdx >= 0 ? S.trials[S.bestIdx].trial_id : -1);
        return `<tr class="${isBest ? 'best-row' : ''}">` +
            cols.map(c => `<td>${fmtCell(getCellValue(t, c))}</td>`).join('') +
            '</tr>';
    }).join('');
}

function getCellValue(trial, col) {
    if (col === 'trial_id') return trial.trial_id;
    if (col === 'rank') return trial.rank;
    if (S.paramNames.includes(col)) return trial.params?.[col] ?? NaN;
    return trial.metrics?.[col] ?? NaN;
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
    if (S.objectives.length === 0) {
        container.innerHTML = '<div style="color:var(--text-2);font-size:0.82rem;padding:8px 0">No objectives configured</div>';
        return;
    }
    container.innerHTML = S.objectives.map((obj, i) => `
    <div class="objective-row">
      <span class="obj-field">${obj.field}</span>
      <span class="obj-type">${obj.obj_type || obj.type || 'minimize'}</span>
      <label style="font-size:0.75rem;color:var(--text-2)">Priority
        <input type="range" min="0" max="5" step="0.1" value="${obj.priority}"
               onchange="S.objectives[${i}].priority=parseFloat(this.value);this.nextElementSibling.textContent=this.value">
        <span style="font-family:var(--mono);color:var(--text-1);min-width:30px">${obj.priority}</span>
      </label>
      <label style="font-size:0.75rem;color:var(--text-2)">Target
        <input type="number" value="${obj.target ?? ''}" step="any"
               onchange="S.objectives[${i}].target=this.value?parseFloat(this.value):null">
      </label>
      <label style="font-size:0.75rem;color:var(--text-2)">Limit
        <input type="number" value="${obj.limit ?? ''}" step="any"
               onchange="S.objectives[${i}].limit=this.value?parseFloat(this.value):null">
      </label>
      <label style="font-size:0.75rem;color:var(--text-2)">Group
        <input type="text" value="${obj.group ?? ''}" style="width:80px;padding:4px 8px;background:var(--bg-2);border:1px solid var(--border);border-radius:4px;color:var(--text-0);font-family:var(--mono);font-size:0.78rem"
               onchange="S.objectives[${i}].group=this.value||null">
      </label>
    </div>
  `).join('');
}

// Client-side TLP rescalarization for preview mode.
async function resetObjectives() {
    S.objectives = JSON.parse(JSON.stringify(S.serverObjectives));
    S.previewActive = false;
    document.getElementById('preview-badge').style.display = 'none';
    if (S.mode === 'live') {
        // Re-fetch trials with the server's actual scores
        const resp = await fetch(`${S.serverUrl}/api/trials?sorted_by=index&include_infeasible=true`);
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
            const raw = m[obj.field];
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
        const resp = await fetch(`${S.serverUrl}/api/objectives`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ objectives: S.objectives }),
        });
        if (!resp.ok) throw new Error('Failed');
        S.previewActive = false;
        document.getElementById('preview-badge').style.display = 'none';
        S.serverObjectives = JSON.parse(JSON.stringify(S.objectives));
        // Re-fetch trials with server-side rescalarization
        const trialsResp = await fetch(`${S.serverUrl}/api/trials?sorted_by=index&include_infeasible=true`);
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
        const resp = await fetch(`${S.serverUrl}/api/checkpoint/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: `Dashboard save at ${new Date().toISOString()}` }),
        });
        const data = await resp.json();
        if (resp.ok) alert(`Checkpoint saved: ${data.path} (${data.trials_saved} trials)`);
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
// Auto-connect from URL params
// ============================================================================
(() => {
    const params = new URLSearchParams(location.search);
    const server = params.get('server');
    if (server) {
        document.getElementById('server-url').value = server;
        connectToServer();
    }
})();

// Update "last trial" timer
setInterval(() => {
    if (S.lastTrialTime) {
        const ago = Math.round((Date.now() - S.lastTrialTime) / 1000);
        document.getElementById('stat-last-time').textContent =
            ago < 60 ? `${ago}s ago` : `${Math.round(ago / 60)}m ago`;
    }
}, 5000);
