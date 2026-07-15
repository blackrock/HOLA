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
    lastEventId: null,        // SSE replay cursor for reconnects
    connectionGeneration: 0, // invalidates superseded connect/resync/load work
    resyncGeneration: null,  // generation currently fetching a full resync
    resyncRetryTimer: null,
    renderScheduled: false,

    trials: [],              // CompletedTrial[] from /api/trials
    trialIndex: new Map(),   // trial_id -> position in trials (O(1) SSE upserts)
    space: [],               // [{name, type, min, max, scale}]
    objectives: [],           // [{field, type, target, limit, priority, group}]
    serverObjectives: [],    // snapshot of objectives as fetched from the server
    paramNames: [],
    metricNames: [],
    metricCounts: new Map(),
    metricExtents: new Map(),
    paramExtents: new Map(),
    scoreGroupCount: 0,
    bestIdx: -1,
    bestRank: Infinity,
    bestScore: Infinity,
    paretoFrontIds: new Set(),
    convergenceScores: [],
    convergenceBest: [],
    convergenceMin: Infinity,
    convergenceMax: -Infinity,
    paretoCache: null,
    tableCache: null,
    sortCol: null,
    sortColumn: null,
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

function sampleEvenly(values, limit) {
    if (values.length <= limit) return values;
    const sampled = [];
    const step = values.length / limit;
    for (let i = 0; i < limit; i++) sampled.push(values[Math.floor(i * step)]);
    return sampled;
}

async function connectToServer() {
    const url = document.getElementById('server-url').value.trim().replace(/\/+$/, '')
        || 'http://localhost:8000';
    document.getElementById('server-url').value = url;

    // A connect attempt supersedes every earlier connect, resync, stream, and
    // in-flight checkpoint read, including another attempt to the same URL.
    const generation = ++S.connectionGeneration;
    S.resyncGeneration = null;
    clearResyncRetry();
    stopStream();
    S.serverUrl = url;
    S.lastEventId = null;

    const tokenField = document.getElementById('api-token');
    if (tokenField && tokenField.value) setApiToken(tokenField.value);

    try {
        // Capture the event watermark before reading the REST snapshot. The
        // stream later reconnects from this cursor, replaying every completion
        // that races the space/objectives/trials requests (duplicates are
        // harmless because trials are upserted by id).
        const cursorResp = await apiFetch(`${url}/api/event_cursor`);
        if (!isCurrentConnection(generation, url)) return;
        if (cursorResp.status === 401) {
            throw new Error('Authentication failed: missing or invalid API token');
        }
        if (!cursorResp.ok) throw new Error('Failed to fetch event cursor');
        const cursorData = await cursorResp.json();
        if (!isCurrentConnection(generation, url)) return;
        const eventCursor = String(cursorData.last_event_id ?? '');
        if (!/^\d+$/.test(eventCursor)) throw new Error('Server returned an invalid event cursor');

        // Probe server by fetching space. All reads go through apiFetch so the
        // bearer token rides along when the server opts into read auth.
        const spaceResp = await apiFetch(`${url}/api/space`);
        if (!isCurrentConnection(generation, url)) return;
        if (spaceResp.status === 401) throw new Error('Authentication failed: missing or invalid API token');
        if (!spaceResp.ok) throw new Error('Server not responding');
        const spaceData = await spaceResp.json();
        if (!isCurrentConnection(generation, url)) return;

        // Fetch objectives
        const objResp = await apiFetch(`${url}/api/objectives`);
        if (!isCurrentConnection(generation, url)) return;
        if (!objResp.ok) throw new Error('Failed to fetch objectives');
        const objData = await objResp.json();
        if (!isCurrentConnection(generation, url)) return;

        // Fetch trials
        const trialsResp = await apiFetch(`${url}/api/trials?sorted_by=index&include_infeasible=true`);
        if (!isCurrentConnection(generation, url)) return;
        if (!trialsResp.ok) throw new Error('Failed to fetch trials');
        const trials = await trialsResp.json();
        if (!isCurrentConnection(generation, url)) return;

        // Commit only after the complete snapshot belongs to the newest
        // connection attempt. This prevents a slow response from an older
        // server from mixing its space/objectives/trials into current state.
        S.space = Array.isArray(spaceData.params) ? spaceData.params : [];
        setParamNames(S.space.map(p => p.name));
        S.objectives = Array.isArray(objData.objectives) ? objData.objectives : [];
        S.serverObjectives = JSON.parse(JSON.stringify(S.objectives));
        replaceTrials(trials);
        S.lastEventId = eventCursor;

        setMode('live');
        renderAll();
        startStream(generation, url);
    } catch (e) {
        if (!isCurrentConnection(generation, url)) return;
        alert('Failed to connect: ' + e.message);
    }
}

function isCurrentConnection(generation, serverUrl) {
    return S.connectionGeneration === generation && S.serverUrl === serverUrl;
}

function isCurrentLiveConnection(generation, serverUrl) {
    return isCurrentConnection(generation, serverUrl) && S.mode === 'live';
}

// Live event stream. EventSource cannot send an Authorization header, so we
// stream /api/events via fetch (which carries the bearer token through
// apiFetch) and parse the text/event-stream body incrementally. This keeps the
// token out of the URL and works whether or not read auth is enabled. The
// stream reconnects on drop.
function startStream(generation = S.connectionGeneration, serverUrl = S.serverUrl) {
    if (!isCurrentLiveConnection(generation, serverUrl)) return;
    stopStream();
    const controller = new AbortController();
    S.sse = controller;
    streamEvents(controller, generation, serverUrl);
}

function stopStream() {
    if (S.sse) {
        S.sse.abort();
        S.sse = null;
    }
}

async function streamEvents(controller, generation, serverUrl) {
    try {
        if (!isCurrentLiveConnection(generation, serverUrl) || S.sse !== controller) return;
        const headers = { Accept: 'text/event-stream' };
        if (S.lastEventId != null) headers['Last-Event-ID'] = String(S.lastEventId);
        const resp = await apiFetch(`${serverUrl}/api/events`, {
            headers,
            signal: controller.signal,
        });
        if (!isCurrentLiveConnection(generation, serverUrl) || S.sse !== controller) return;
        if (!resp.ok || !resp.body) {
            setDot('disconnected');
            scheduleReconnect(controller, generation, serverUrl);
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
                handleStreamEvent(chunk, generation, serverUrl);
                if (S.sse !== controller) return;
            }
        }
        // Stream ended cleanly; reconnect unless this stream was superseded.
        if (!isCurrentLiveConnection(generation, serverUrl) || S.sse !== controller) return;
        setDot('disconnected');
        scheduleReconnect(controller, generation, serverUrl);
    } catch (e) {
        if (controller.signal.aborted) return; // intentionally stopped
        if (!isCurrentLiveConnection(generation, serverUrl) || S.sse !== controller) return;
        setDot('disconnected');
        scheduleReconnect(controller, generation, serverUrl);
    }
}

function scheduleReconnect(controller, generation, serverUrl) {
    // Only reconnect if this controller is still the active stream.
    if (S.sse !== controller || !isCurrentLiveConnection(generation, serverUrl)) return;
    setTimeout(() => {
        if (S.sse === controller && isCurrentLiveConnection(generation, serverUrl)) {
            streamEvents(controller, generation, serverUrl);
        }
    }, 2000);
}

function clearResyncRetry() {
    if (S.resyncRetryTimer != null) {
        clearTimeout(S.resyncRetryTimer);
        S.resyncRetryTimer = null;
    }
}

function scheduleResyncRetry(generation, serverUrl) {
    if (S.resyncRetryTimer != null || !isCurrentLiveConnection(generation, serverUrl)) return;
    S.resyncRetryTimer = setTimeout(() => {
        S.resyncRetryTimer = null;
        if (isCurrentLiveConnection(generation, serverUrl)) {
            void resyncLiveState(generation, serverUrl);
        }
    }, 1000);
}

// Parse one text/event-stream record (lines separated by \n) and dispatch its
// JSON data payload to handleEngineEvent.
function handleStreamEvent(
    chunk,
    generation = S.connectionGeneration,
    serverUrl = S.serverUrl,
) {
    if (!isCurrentLiveConnection(generation, serverUrl)) return;
    const dataLines = [];
    let eventType = 'message';
    let eventId = null;
    for (const line of chunk.split('\n')) {
        if (line.startsWith('data:')) {
            // Strip the "data:" prefix and a single optional leading space.
            dataLines.push(line.slice(line[5] === ' ' ? 6 : 5));
        } else if (line.startsWith('event:')) {
            eventType = line.slice(line[6] === ' ' ? 7 : 6);
        } else if (line.startsWith('id:')) {
            eventId = line.slice(line[3] === ' ' ? 4 : 3);
        }
        // Lines starting with ':' are keep-alive comments; ignore them.
    }
    if (eventId !== null && /^\d+$/.test(eventId)) S.lastEventId = eventId;
    if (eventType === 'stream_lagged' || eventType === 'stream_reset') {
        void resyncLiveState(generation, serverUrl);
        return;
    }
    if (dataLines.length === 0) return;
    let event;
    try {
        event = JSON.parse(dataLines.join('\n'));
    } catch {
        return;
    }
    void handleEngineEvent(event, generation, serverUrl);
}

async function resyncLiveState(
    generation = S.connectionGeneration,
    serverUrl = S.serverUrl,
) {
    if (!isCurrentLiveConnection(generation, serverUrl)) return;
    if (S.resyncGeneration === generation) return;
    S.resyncGeneration = generation;
    clearResyncRetry();
    // Freeze event application while taking the replacement snapshot. The
    // cursor captured first is replayed after the snapshot is committed, so an
    // event racing any REST request cannot be overwritten or lost.
    stopStream();
    let resynced = false;
    try {
        const cursorResp = await apiFetch(`${serverUrl}/api/event_cursor`);
        if (!isCurrentLiveConnection(generation, serverUrl)) return;
        if (!cursorResp.ok) throw new Error('Failed to fetch event cursor during resync');
        const cursorData = await cursorResp.json();
        if (!isCurrentLiveConnection(generation, serverUrl)) return;
        const eventCursor = String(cursorData.last_event_id ?? '');
        if (!/^\d+$/.test(eventCursor)) throw new Error('Invalid event cursor during resync');

        const [spaceResp, objectivesResp, trialsResp] = await Promise.all([
            apiFetch(`${serverUrl}/api/space`),
            apiFetch(`${serverUrl}/api/objectives`),
            apiFetch(`${serverUrl}/api/trials?sorted_by=index&include_infeasible=true`),
        ]);
        if (!isCurrentLiveConnection(generation, serverUrl)) return;
        if (!spaceResp.ok || !objectivesResp.ok || !trialsResp.ok) {
            throw new Error('Failed to fetch live snapshot during resync');
        }
        const [spaceData, objectivesData, trials] = await Promise.all([
            spaceResp.json(), objectivesResp.json(), trialsResp.json(),
        ]);
        // Ignore a late response after any newer connection or checkpoint
        // load, even when it targets the same URL.
        if (!isCurrentLiveConnection(generation, serverUrl)) return;
        S.space = Array.isArray(spaceData.params) ? spaceData.params : [];
        setParamNames(S.space.map(p => p.name));
        S.objectives = Array.isArray(objectivesData.objectives)
            ? objectivesData.objectives : [];
        S.serverObjectives = JSON.parse(JSON.stringify(S.objectives));
        replaceTrials(trials);
        S.lastEventId = eventCursor;
        renderAll();
        resynced = true;
    } catch {
        if (isCurrentLiveConnection(generation, serverUrl)) setDot('disconnected');
    } finally {
        // A stale request must not clear the in-progress marker for a newer
        // generation's resync.
        if (S.resyncGeneration === generation) S.resyncGeneration = null;
        if (isCurrentLiveConnection(generation, serverUrl)) {
            if (resynced) startStream(generation, serverUrl);
            else scheduleResyncRetry(generation, serverUrl);
        }
    }
}

async function handleEngineEvent(
    event,
    generation = S.connectionGeneration,
    serverUrl = S.serverUrl,
) {
    if (!isCurrentLiveConnection(generation, serverUrl)) return;
    if (event.type === 'ObjectivesChanged') {
        await resyncLiveState(generation, serverUrl);
    } else if (event.type === 'TrialCompleted') {
        const trial = event.trial || await fetchCompletedTrial(event.trial_id, serverUrl);
        if (!isCurrentLiveConnection(generation, serverUrl)) return;
        if (!trial) return;
        if (S.previewActive) rescoreTrialForPreview(trial);
        upsertTrial(trial);
        S.lastTrialTime = Date.now();
        scheduleRenderAll();
    }
}

async function fetchCompletedTrial(trialId, serverUrl = S.serverUrl) {
    const resp = await apiFetch(`${serverUrl}/api/trial/${trialId}?include_infeasible=true`);
    if (!resp.ok) return null;
    return resp.json();
}

function setParamNames(names) {
    S.paramNames = Array.isArray(names) ? names : [];
}

function scoreGroupWidth(trial) {
    const vector = trial?.score_vector;
    return vector && typeof vector === 'object' && !Array.isArray(vector)
        ? Object.keys(vector).length : 0;
}

function recordFiniteExtent(extents, name, value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return;
    const current = extents.get(name);
    if (current) {
        if (value < current.min) current.min = value;
        if (value > current.max) current.max = value;
    } else {
        extents.set(name, { min: value, max: value });
    }
}

// Update the small field-name/extents indexes for one newly appended trial.
// Returns true when the visible metric column set changed.
function recordTrialFields(trial) {
    let newMetric = false;
    const metrics = trial?.metrics;
    if (metrics && typeof metrics === 'object' && !Array.isArray(metrics)) {
        for (const [name, raw] of Object.entries(metrics)) {
            const previous = S.metricCounts.get(name) || 0;
            S.metricCounts.set(name, previous + 1);
            if (previous === 0) newMetric = true;
            const value = raw === 'inf' ? Infinity : raw === '-inf' ? -Infinity : raw;
            recordFiniteExtent(S.metricExtents, name, value);
        }
    }
    const params = trial?.params;
    if (params && typeof params === 'object' && !Array.isArray(params)) {
        for (const [name, value] of Object.entries(params)) {
            recordFiniteExtent(S.paramExtents, name, value);
        }
    }
    return newMetric;
}

function appendConvergenceTrial(trial, multiFrontCount = null) {
    if (isSingleObjective()) {
        const score = getTrialScore(trial);
        const value = Number.isFinite(score) ? score : null;
        S.convergenceScores.push(value);
        const previous = S.convergenceBest.length === 0
            ? null : S.convergenceBest[S.convergenceBest.length - 1];
        const running = value === null
            ? previous : previous === null ? value : Math.min(previous, value);
        S.convergenceBest.push(running);
        if (value !== null) {
            if (value < S.convergenceMin) S.convergenceMin = value;
            if (value > S.convergenceMax) S.convergenceMax = value;
        }
        return;
    }

    S.convergenceScores.push(null);
    const frontCount = multiFrontCount ?? S.paretoFrontIds.size;
    S.convergenceBest.push(frontCount);
    if (frontCount < S.convergenceMin) S.convergenceMin = frontCount;
    if (frontCount > S.convergenceMax) S.convergenceMax = frontCount;
}

function rebuildBestAndConvergence() {
    S.bestIdx = -1;
    S.bestRank = Infinity;
    S.bestScore = Infinity;
    S.paretoFrontIds = new Set();
    S.convergenceScores = [];
    S.convergenceBest = [];
    S.convergenceMin = Infinity;
    S.convergenceMax = -Infinity;
    if (isSingleObjective()) {
        for (let i = 0; i < S.trials.length; i++) {
            const score = getTrialScore(S.trials[i]);
            if (Number.isFinite(score) && score < S.bestScore) {
                S.bestScore = score;
                S.bestIdx = i;
            }
        }
        if (S.bestIdx >= 0) {
            S.bestRank = 0;
            S.paretoFrontIds.add(S.trials[S.bestIdx].trial_id);
        }
    } else {
        for (let i = 0; i < S.trials.length; i++) {
            const trial = S.trials[i];
            if (trial.pareto_front === 0 && isTrialFeasible(trial)) {
                S.paretoFrontIds.add(trial.trial_id);
                if (S.bestIdx < 0) S.bestIdx = i;
            }
        }
        if (S.bestIdx >= 0) S.bestRank = 0;
    }

    let encounteredFront = 0;
    for (const trial of S.trials) {
        if (!isSingleObjective() && S.paretoFrontIds.has(trial.trial_id)) encounteredFront++;
        appendConvergenceTrial(trial, isSingleObjective() ? null : encounteredFront);
    }
}

function markTrialOffFront(trial) {
    trial.pareto_front = Math.max(1, Number.isFinite(trial.pareto_front)
        ? trial.pareto_front : 1);
    if (!Number.isFinite(trial.rank) || trial.rank === 0) trial.rank = 1;
}

function scoreVectorDominates(a, b) {
    const aVector = a?.score_vector;
    const bVector = b?.score_vector;
    if (!isTrialFeasible(a) || !isTrialFeasible(b)) return false;
    const aKeys = Object.keys(aVector);
    if (aKeys.length !== Object.keys(bVector).length) return false;
    let strictly = false;
    for (const key of aKeys) {
        if (!Object.prototype.hasOwnProperty.call(bVector, key)) return false;
        if (aVector[key] > bVector[key]) return false;
        if (aVector[key] < bVector[key]) strictly = true;
    }
    return strictly;
}

function updateScalarBestForAppend(trial, idx) {
    const demoted = [];
    const score = getTrialScore(trial);
    if (Number.isFinite(score) && score < S.bestScore) {
        if (S.bestIdx >= 0) {
            const previous = S.trials[S.bestIdx];
            S.paretoFrontIds.delete(previous.trial_id);
            markTrialOffFront(previous);
            demoted.push(previous);
        }
        S.bestIdx = idx;
        S.bestRank = 0;
        S.bestScore = score;
        S.paretoFrontIds.add(trial.trial_id);
        trial.rank = 0;
        trial.pareto_front = 0;
    } else {
        S.paretoFrontIds.delete(trial.trial_id);
        markTrialOffFront(trial);
    }
    return demoted;
}

function chooseFirstParetoTrial() {
    let first = -1;
    for (const trialId of S.paretoFrontIds) {
        const idx = S.trialIndex.get(trialId);
        if (idx !== undefined && (first < 0 || idx < first)) first = idx;
    }
    S.bestIdx = first;
    S.bestRank = first >= 0 ? 0 : Infinity;
}

function updateMultiObjectiveFrontForAppend(trial) {
    const demoted = [];
    if (!isTrialFeasible(trial)) {
        S.paretoFrontIds.delete(trial.trial_id);
        markTrialOffFront(trial);
        return demoted;
    }

    let isDominated = false;
    for (const frontId of S.paretoFrontIds) {
        const frontIdx = S.trialIndex.get(frontId);
        const frontTrial = frontIdx === undefined ? null : S.trials[frontIdx];
        if (!frontTrial) continue;
        if (scoreVectorDominates(frontTrial, trial)) {
            isDominated = true;
            break;
        }
        if (scoreVectorDominates(trial, frontTrial)) demoted.push(frontTrial);
    }

    if (isDominated) {
        markTrialOffFront(trial);
        return [];
    }
    for (const oldFront of demoted) {
        S.paretoFrontIds.delete(oldFront.trial_id);
        markTrialOffFront(oldFront);
    }
    trial.pareto_front = 0;
    S.paretoFrontIds.add(trial.trial_id);
    chooseFirstParetoTrial();
    return demoted;
}

// Replace a REST/checkpoint snapshot and rebuild all indexes once. Live SSE
// appends use upsertTrial below and do not pay this O(n) setup cost again.
function replaceTrials(trials) {
    S.trials = Array.isArray(trials) ? trials : [];
    S.trialIndex = new Map();
    S.metricCounts = new Map();
    S.metricExtents = new Map();
    S.paramExtents = new Map();
    S.scoreGroupCount = 0;
    for (let i = 0; i < S.trials.length; i++) {
        const trial = S.trials[i];
        S.trialIndex.set(trial.trial_id, i);
        recordTrialFields(trial);
        S.scoreGroupCount = Math.max(S.scoreGroupCount, scoreGroupWidth(trial));
    }
    S.metricNames = [...S.metricCounts.keys()].sort();
    S.paretoCache = null;
    S.tableCache = null;
    reconcileTableSort();
    rebuildBestAndConvergence();
}

function sameTrialPayload(a, b) {
    if (Object.is(a, b)) return true;
    if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length) return false;
    for (const key of aKeys) {
        if (!Object.prototype.hasOwnProperty.call(b, key) || !sameTrialPayload(a[key], b[key])) {
            return false;
        }
    }
    return true;
}

function upsertTrial(trial) {
    const existing = S.trialIndex.get(trial.trial_id);
    if (existing !== undefined) {
        const current = S.trials[existing];
        // The snapshot watermark deliberately allows an event already present
        // in the REST snapshot to replay. Ignore an identical payload without
        // invalidating any large-history cache.
        if (sameTrialPayload(current, trial)) return false;
        // Changed replacements are uncommon and may alter any cached field.
        // The ID lookup is O(1); rebuilding keeps replacements fully correct.
        S.trials[existing] = trial;
        replaceTrials(S.trials);
        return false;
    }

    const previousSingle = isSingleObjective();
    const idx = S.trials.length;
    S.trials.push(trial);
    S.trialIndex.set(trial.trial_id, idx);
    if (recordTrialFields(trial)) S.metricNames = [...S.metricCounts.keys()].sort();
    S.scoreGroupCount = Math.max(S.scoreGroupCount, scoreGroupWidth(trial));

    // A change between scalar and vector semantics invalidates the shape of the
    // whole convergence series. In the steady state, an append only extends it.
    const changedObjectiveShape = previousSingle !== isSingleObjective();
    let demoted = [];
    if (changedObjectiveShape) {
        rebuildBestAndConvergence();
        S.paretoCache = null;
    } else if (isSingleObjective()) {
        demoted = updateScalarBestForAppend(trial, idx);
        appendConvergenceTrial(trial);
    } else {
        demoted = updateMultiObjectiveFrontForAppend(trial);
        appendConvergenceTrial(trial);
    }
    if (!changedObjectiveShape) updateParetoCacheForAppend(trial, idx, demoted);
    if (demoted.length > 0 && S.sortColumn?.key === 'builtin:rank') S.tableCache = null;
    else updateTableCacheForAppend(trial, idx);
    return true;
}

// readAsText necessarily buffers the JSON in memory. A 64 MiB ceiling keeps
// that bounded while leaving comfortable headroom for the supported 100k-trial
// release gate (a representative export is already larger than the old 8 MiB).
const MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024;

function isCheckpointFileSizeAllowed(size) {
    return Number.isFinite(size) && size >= 0 && size <= MAX_CHECKPOINT_BYTES;
}

function loadCheckpointFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    // Guard against oversized files: readAsText buffers the whole file into a
    // string on the main thread, so a multi-hundred-MB drop would hang the tab
    // or exhaust memory. Reject anything beyond a reasonable checkpoint size.
    if (!isCheckpointFileSizeAllowed(file.size)) {
        alert(`Checkpoint file is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). ` +
            `Maximum supported size is ${MAX_CHECKPOINT_BYTES / (1024 * 1024)} MB.`);
        event.target.value = '';
        return;
    }
    // Tear down any live stream and invalidate in-flight connects/resyncs so
    // their late responses cannot bleed into the offline snapshot.
    const generation = ++S.connectionGeneration;
    S.resyncGeneration = null;
    clearResyncRetry();
    stopStream();
    const reader = new FileReader();
    reader.onload = (e) => {
        if (S.connectionGeneration !== generation) return;
        try {
            const data = JSON.parse(e.target.result);
            if (S.connectionGeneration !== generation) return;
            applyCheckpointData(data);
        } catch (err) {
            if (S.connectionGeneration !== generation) return;
            alert('Failed to parse checkpoint: ' + err.message);
        }
    };
    reader.readAsText(file);
    event.target.value = ''; // Reset so same file can be loaded again
}

// Convert the StudyConfig map embedded in a full checkpoint to the dashboard's
// array representation. Keeping the declared type/choices is essential for
// categorical axes; inferring them from the first candidate loses that data.
function spaceFromStudyConfig(config) {
    const space = config?.space;
    if (!space || typeof space !== 'object' || Array.isArray(space)) return [];
    return Object.entries(space)
        .filter(([, param]) => param && typeof param === 'object' && !Array.isArray(param))
        .map(([name, param]) => ({ ...param, name }));
}

function checkpointDataToDashboard(data) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        throw new Error('Invalid checkpoint format');
    }

    let trials;
    let space;
    let objectives;
    if (data.format === 'hola-dashboard-export') {
        if (!Array.isArray(data.trials)) throw new Error('Invalid dashboard export');
        trials = data.trials;
        space = Array.isArray(data.space) ? data.space : [];
        objectives = Array.isArray(data.objectives) ? data.objectives : [];
    } else {
        // Support full Checkpoint wrappers and leaderboard-only files.
        const lb = (data.checkpoint || data).leaderboard;
        if (!lb || !Array.isArray(lb.trials)) throw new Error('Invalid checkpoint format');
        // Map persisted trials to the dashboard's CompletedTrial view.
        trials = lb.trials.map((t, i) => ({
            trial_id: t.trial_id ?? t.id ?? i,
            params: t.candidate ?? t.params ?? {},
            metrics: t.raw_metrics ?? t.metrics ?? {},
            scores: t.scores ?? {},
            score_vector: t.score_vector ?? observationToScoreVector(t.observation),
            rank: t.rank,
            pareto_front: t.pareto_front,
            completed_at: t.timestamp ?? t.completed_at ?? 0,
        }));
        space = spaceFromStudyConfig(data.config);
        objectives = Array.isArray(data.config?.objectives) ? data.config.objectives : [];
    }

    computeRanksIfMissing(trials);
    let paramNames = space.map(p => p.name);
    // Infer parameter metadata only when a checkpoint did not carry it.
    if (space.length === 0 && trials.length > 0 && trials[0].params) {
        const c = typeof trials[0].params === 'object' ? trials[0].params : {};
        paramNames = Object.keys(c);
        space = paramNames.map(name => ({
            name, type: 'real', min: 0, max: 1, scale: 'linear'
        }));
        // Compute actual bounds from data
        for (const p of space) {
            const vals = trials.map(t => t.params?.[p.name]).filter(v => v != null);
            if (vals.length > 0) {
                const mm = minMax(vals);
                p.min = mm.min;
                p.max = mm.max;
            }
        }
    }
    return { trials, space, objectives, paramNames };
}

function applyCheckpointData(data) {
    const snapshot = checkpointDataToDashboard(data);
    S.space = snapshot.space;
    setParamNames(snapshot.paramNames);
    S.objectives = snapshot.objectives;
    S.serverObjectives = JSON.parse(JSON.stringify(S.objectives));
    S.previewActive = false;
    const badge = document.getElementById('preview-badge');
    if (badge) badge.style.display = 'none';
    replaceTrials(snapshot.trials);
    setMode('offline');
    renderAll();
}

// Convert a persisted leaderboard observation into the dashboard's score_vector
// shape ({group: score}). The engine stores either a scalar (single objective)
// or a {group: score} map (multi-group). Anything else yields an empty vector.
function observationToScoreVector(observation) {
    const decoded = decodePersistedValue(observation);
    if (typeof decoded === 'number') return { score: decoded };
    if (decoded === 'inf') return { score: Infinity };
    if (decoded === '-inf') return { score: -Infinity };
    if (decoded && typeof decoded === 'object') return { ...decoded };
    return {};
}

function decodePersistedValue(value) {
    if (Array.isArray(value)) return value.map(decodePersistedValue);
    if (!value || typeof value !== 'object') return value;
    if (Object.keys(value).length === 1 && typeof value['$hola.float'] === 'string') {
        const tag = value['$hola.float'].replace(/^f(?:32|64):/, '');
        if (tag === '+inf') return Infinity;
        if (tag === '-inf') return -Infinity;
        if (tag === 'nan') return NaN;
    }
    if (Object.keys(value).length === 1 && value['$hola.map']) {
        return decodePersistedValue(value['$hola.map']);
    }
    return Object.fromEntries(
        Object.entries(value).map(([key, item]) => [key, decodePersistedValue(item)])
    );
}

function twoObjectiveFrontRanks(entries, cols) {
    const points = entries.map((entry, position) => {
        const xRaw = entry.trial.score_vector[cols[0]];
        const yRaw = entry.trial.score_vector[cols[1]];
        return {
            position,
            x: xRaw === 0 ? 0 : xRaw,
            y: yRaw === 0 ? 0 : yRaw,
        };
    });
    points.sort((a, b) => a.x - b.x || a.y - b.y || a.position - b.position);
    const yValues = [...new Set(points.map(point => point.y).sort((a, b) => a - b))];
    const tree = new Array(yValues.length + 1).fill(0);
    const query = index => {
        let best = 0;
        while (index > 0) {
            best = Math.max(best, tree[index]);
            index &= index - 1;
        }
        return best;
    };
    const update = (index, value) => {
        while (index < tree.length) {
            tree[index] = Math.max(tree[index], value);
            index += index & -index;
        }
    };
    const yIndexOf = value => {
        let low = 0, high = yValues.length;
        while (low < high) {
            const middle = (low + high) >>> 1;
            if (yValues[middle] < value) low = middle + 1;
            else high = middle;
        }
        return low + 1;
    };

    const fronts = new Array(entries.length).fill(0);
    let start = 0;
    while (start < points.length) {
        let end = start + 1;
        while (end < points.length
            && points[end].x === points[start].x
            && points[end].y === points[start].y) end++;
        const yIndex = yIndexOf(points[start].y);
        const front = query(yIndex);
        for (let i = start; i < end; i++) fronts[points[i].position] = front;
        update(yIndex, front + 1);
        start = end;
    }
    return fronts;
}

function exactFrontRanksBounded(entries, cols) {
    const n = entries.length;
    const dominationCount = new Array(n).fill(0);
    const dominated = Array.from({ length: n }, () => []);
    const dominates = (a, b) => {
        let strictly = false;
        for (const col of cols) {
            const av = entries[a].trial.score_vector[col];
            const bv = entries[b].trial.score_vector[col];
            if (av > bv) return false;
            if (av < bv) strictly = true;
        }
        return strictly;
    };
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            if (dominates(i, j)) {
                dominated[i].push(j);
                dominationCount[j]++;
            } else if (dominates(j, i)) {
                dominated[j].push(i);
                dominationCount[i]++;
            }
        }
    }
    let current = [];
    for (let i = 0; i < n; i++) if (dominationCount[i] === 0) current.push(i);
    const fronts = new Array(n).fill(1);
    let front = 0;
    while (current.length > 0) {
        const next = [];
        for (const i of current) {
            fronts[i] = front;
            for (const j of dominated[i]) {
                dominationCount[j]--;
                if (dominationCount[j] === 0) next.push(j);
            }
        }
        current = next;
        front++;
    }
    return fronts;
}

// For large 3+-group imports, exact non-dominated sorting is quadratic. Keep a
// safe under-approximation instead: the lexicographic minimum is guaranteed to
// be truly non-dominated, while every unverified point is conservatively placed
// behind front zero.
function conservativeManyObjectiveFrontRanks(entries, cols) {
    const fronts = new Array(entries.length).fill(1);
    if (entries.length === 0) return fronts;
    let best = 0;
    for (let i = 1; i < entries.length; i++) {
        let cmp = 0;
        for (const col of cols) {
            const a = entries[i].trial.score_vector[col];
            const b = entries[best].trial.score_vector[col];
            if (a < b) { cmp = -1; break; }
            if (a > b) { cmp = 1; break; }
        }
        if (cmp < 0 || (cmp === 0 && entries[i].index < entries[best].index)) best = i;
    }
    fronts[best] = 0;
    return fronts;
}

// Compute rank and pareto_front client-side for any trial that lacks them (a
// real persisted leaderboard has neither). Defaulting rank to the insertion
// index would always flag trial 0 as best, so we rank off the derived scores
// instead. Valid server-provided ranks are left untouched; malformed or stale
// offline ranks are repaired.
function computeRanksIfMissing(trials) {
    const hasRank = t => Number.isFinite(t.rank) && t.rank >= 0;
    const hasParetoFront = t => Number.isFinite(t.pareto_front) && t.pareto_front >= 0;
    const needs = trials.filter(t => !hasRank(t) || !hasParetoFront(t));

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
        if (needs.length === 0) return;
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
            if (!hasRank(entry.t)) entry.t.rank = rank;
            if (!hasParetoFront(entry.t)) entry.t.pareto_front = entry.t.rank;
        });
        return;
    }

    // Multi-group case: exact O(n log n) two-objective ranks, exact bounded
    // sorting for small higher-dimensional imports, and a conservative front
    // for large 3+-group imports. A trial is eligible only when every group has
    // a finite numeric score; malformed/infeasible observations never reach
    // front zero.
    const cols = [...groupNames];
    const hasFiniteVector = t => cols.every(g => Number.isFinite(scoreVal(t, g)));
    const feasible = [];
    const infeasible = [];
    for (let index = 0; index < trials.length; index++) {
        const entry = { trial: trials[index], index };
        if (hasFiniteVector(entry.trial)) feasible.push(entry);
        else infeasible.push(entry.trial);
    }

    const EXACT_MANY_OBJECTIVE_LIMIT = 2048;
    let fronts;
    if (cols.length === 2) {
        fronts = twoObjectiveFrontRanks(feasible, cols);
    } else if (feasible.length <= EXACT_MANY_OBJECTIVE_LIMIT) {
        fronts = exactFrontRanksBounded(feasible, cols);
    } else {
        fronts = conservativeManyObjectiveFrontRanks(feasible, cols);
    }
    const order = feasible
        .map((entry, position) => ({ entry, position, front: fronts[position] }))
        .sort((a, b) => a.front - b.front || a.entry.index - b.entry.index);
    let worstFeasibleRank = 0;
    let worstFeasibleFront = 0;
    order.forEach((ranked, rank) => {
        const trial = ranked.entry.trial;
        if (!hasRank(trial)) trial.rank = rank;
        if (!hasParetoFront(trial)) trial.pareto_front = ranked.front;
        worstFeasibleRank = Math.max(worstFeasibleRank, trial.rank);
        worstFeasibleFront = Math.max(worstFeasibleFront, trial.pareto_front);
    });

    // Keep every infeasible vector strictly behind all feasible fronts. Also
    // repair rank-0 values from older dashboard exports produced by this bug.
    const infeasibleRank = Math.max(1, worstFeasibleRank + 1);
    const infeasibleFront = Math.max(1, worstFeasibleFront + 1);
    for (const t of infeasible) {
        if (!Number.isFinite(t.rank) || t.rank < infeasibleRank) t.rank = infeasibleRank;
        if (!Number.isFinite(t.pareto_front) || t.pareto_front < infeasibleFront) {
            t.pareto_front = infeasibleFront;
        }
    }
}

// ============================================================================
// Helpers for score extraction
// ============================================================================

/// True when every trial carries a single comparable scalar score, i.e. there
/// is a single objective group. Summing across multiple groups would collapse
/// incomparable Pareto axes into a meaningless scalar, so callers must treat the
/// multi-objective case via rank / pareto_front instead.
function isSingleObjective() {
    return S.scoreGroupCount === 1;
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

// The server and persisted checkpoints encode constraint violations as a
// non-finite component (usually the string sentinel "inf"). Never treat such
// a trial as Pareto-optimal, even if an older server/export supplied rank 0.
function isTrialFeasible(trial) {
    const vector = trial.score_vector;
    if (!vector || typeof vector !== 'object' || Array.isArray(vector)) return false;
    let count = 0;
    for (const value of Object.values(vector)) {
        count++;
        if (typeof value !== 'number' || !Number.isFinite(value)) return false;
    }
    return count > 0;
}

/// The authoritative "best" trial. The server provides rank and pareto_front;
/// the rank-0 / pareto_front-0 trial is the best (or a Pareto-optimal trial in
/// the multi-objective case) without inventing a cross-axis scalar.
function getTrialRank(trial) {
    if (!isTrialFeasible(trial)) return Infinity;
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

function scheduleRenderAll() {
    if (S.renderScheduled) return;
    S.renderScheduled = true;
    requestAnimationFrame(() => {
        S.renderScheduled = false;
        renderAll();
    });
}

function updateStats() {
    const best = findBest();
    document.getElementById('stat-trials').textContent = S.trials.length;
    // For a single objective the scalar score is meaningful; for multiple
    // objectives report a current-front trial id instead of a fabricated sum.
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

// Scalar best follows the oriented server score so a later rank-0 completion
// can replace an earlier stale rank-0 DTO. Multi-objective best is the earliest
// member of the incrementally maintained current frontier.
function findBest() {
    return S.bestIdx >= 0 ? S.trials[S.bestIdx] : null;
}

// ============================================================================
// Convergence Chart (dependency-free canvas)
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
    const canvas = document.getElementById('convergence-canvas');
    const ctx = canvas.getContext('2d');
    const card = container.closest('.card');
    const cardW = card ? card.clientWidth - 34 : 0;
    const containerW = container.clientWidth;
    const w = Math.max(cardW, containerW, 280);
    const h = 280;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (S.trials.length === 0) return;
    const single = isSingleObjective();

    // These arrays and their bounds are maintained when snapshots/appends are
    // committed, so a repaint never rebuilds convergence from full history.
    const ys = S.convergenceScores;
    const runBest = S.convergenceBest;
    const bestLabel = single ? 'Best' : 'Pareto front size';
    if (!Number.isFinite(S.convergenceMin) || !Number.isFinite(S.convergenceMax)) return;
    let yMin = S.convergenceMin, yMax = S.convergenceMax;
    const useLog = yMin > 0 && yMax / yMin > 1000;
    const transformY = useLog ? (v => Math.log10(v)) : (v => v);
    yMin = transformY(yMin);
    yMax = transformY(yMax);
    const yPad = (yMax - yMin) * 0.05 || 1;
    yMin -= yPad;
    yMax += yPad;

    const pad = { top: 25, right: 18, bottom: 34, left: 62 };
    const pw = Math.max(1, w - pad.left - pad.right);
    const ph = Math.max(1, h - pad.top - pad.bottom);
    const sx = index => pad.left + index / Math.max(1, S.trials.length - 1) * pw;
    const sy = value => pad.top + ph - (transformY(value) - yMin) / (yMax - yMin) * ph;

    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#8c8cbc';
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const x = pad.left + pw * i / 4;
        const y = pad.top + ph * i / 4;
        ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + ph); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
        ctx.textAlign = 'center';
        ctx.fillText(String(Math.round((S.trials.length - 1) * i / 4)), x, h - 10);
        const transformedTick = yMax - (yMax - yMin) * i / 4;
        const tick = useLog ? 10 ** transformedTick : transformedTick;
        ctx.textAlign = 'right';
        ctx.fillText(fmt(tick), pad.left - 7, y + 4);
    }

    function drawSeries(values, color, width, dashed) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dashed ? [6, 3] : []);
        ctx.beginPath();
        let drawing = false;
        const sampleCount = Math.min(values.length, 5000);
        const step = values.length / Math.max(1, sampleCount);
        for (let sample = 0; sample < sampleCount; sample++) {
            const index = values.length <= sampleCount ? sample : Math.floor(sample * step);
            const value = values[index];
            if (value === null || !isFinite(value) || (useLog && value <= 0)) {
                drawing = false;
                continue;
            }
            const x = sx(index);
            const y = sy(value);
            if (drawing) ctx.lineTo(x, y); else ctx.moveTo(x, y);
            drawing = true;
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    if (single) drawSeries(ys, '#6c5ce7', 1.5, false);
    drawSeries(runBest, '#00cec9', 2, single);

    ctx.textAlign = 'left';
    ctx.fillStyle = '#00cec9';
    ctx.fillText(bestLabel + (useLog ? ' (log scale)' : ''), pad.left, 14);
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

    // Build this bounded cache once per axis pair. Subsequent SSE appends update
    // its exact bounds and reservoirs in O(1), so repainting does not rescan the
    // full study history.
    const pareto = getParetoRenderCache(xField, yField);
    const { front, dominated } = pareto;
    let { xMin, xMax, yMin, yMax } = pareto;
    if (!Number.isFinite(xMin) || !Number.isFinite(yMin)) return;

    // Scale
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
    ctx.fillStyle = '#8c8cbc';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(xField, pad.left + pw / 2, h - 5);
    ctx.save();
    ctx.translate(12, pad.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yField, 0, 0);
    ctx.restore();

    // Axis tick labels
    ctx.fillStyle = '#8c8cbc';
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
    const points = front.concat(dominated);
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
            trial: p.trial,
            xVal: p.x, yVal: p.y,
            xField, yField,
            onFront: p.onFront,
        });
    }

    // Attach tooltip listener (once)
    attachParetoTooltip();
}

// Deterministic reservoir sampling keeps memory bounded without making chart
// output depend on Math.random (important for reproducible screenshots/tests).
function addBoundedSample(sample, item, seen, limit) {
    if (sample.length < limit) {
        sample.push(item);
        return;
    }
    const slot = (Math.imul(seen, 2654435761) >>> 0) % seen;
    if (slot < limit) sample[slot] = item;
}

function addTrialToParetoCache(cache, trial) {
    const x = getMetric(trial, cache.xField);
    const y = getMetric(trial, cache.yField);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    if (x < cache.xMin) cache.xMin = x;
    if (x > cache.xMax) cache.xMax = x;
    if (y < cache.yMin) cache.yMin = y;
    if (y > cache.yMax) cache.yMax = y;
    const point = {
        x, y, trial,
        onFront: S.paretoFrontIds.has(trial.trial_id) && isTrialFeasible(trial),
    };
    if (point.onFront) {
        cache.frontSeen++;
        addBoundedSample(cache.front, point, cache.frontSeen, 2500);
    } else {
        cache.dominatedSeen++;
        addBoundedSample(cache.dominated, point, cache.dominatedSeen, 2500);
    }
}

function getParetoRenderCache(xField, yField) {
    const key = `${xField}\u0000${yField}`;
    if (S.paretoCache?.key === key && S.paretoCache.sourceLength === S.trials.length) {
        return S.paretoCache;
    }
    const cache = {
        key, xField, yField,
        sourceLength: S.trials.length,
        front: [], dominated: [],
        frontSeen: 0, dominatedSeen: 0,
        xMin: Infinity, xMax: -Infinity,
        yMin: Infinity, yMax: -Infinity,
    };
    for (const trial of S.trials) addTrialToParetoCache(cache, trial);
    S.paretoCache = cache;
    return cache;
}

function updateParetoCacheForAppend(trial, index, demoted = []) {
    const cache = S.paretoCache;
    if (!cache) return;
    if (cache.sourceLength !== index) {
        S.paretoCache = null;
        return;
    }
    for (const oldFront of demoted) addTrialToParetoCache(cache, oldFront);
    addTrialToParetoCache(cache, trial);
    cache.sourceLength++;

    // Front membership can lose old points when a late trial dominates them.
    // Re-sample only the current frontier, never the full history.
    cache.front = [];
    cache.frontSeen = 0;
    for (const trialId of S.paretoFrontIds) {
        const frontIdx = S.trialIndex.get(trialId);
        if (frontIdx !== undefined) addTrialToParetoCache(cache, S.trials[frontIdx]);
    }
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
    const displayTrials = sampleEvenly(S.trials, 2000);
    const n = names.length;
    const gap = pw / (n - 1);

    // Compute min/max per param, with categorical support
    const ranges = names.map(name => {
        const sp = S.space.find(s => s.name === name);
        if (sp && sp.type === 'categorical' && sp.choices) {
            return { min: 0, max: sp.choices.length - 1, categorical: true, choices: sp.choices };
        }
        if (sp) return { min: sp.min, max: sp.max };
        return S.paramExtents.get(name) || { min: NaN, max: NaN };
    });

    // Coloring metric: the scalar score when single-objective, otherwise the
    // server-provided rank (lower is better) so multi-objective lines still
    // convey relative quality without an invalid cross-axis scalar.
    const single = isSingleObjective();
    const colorValue = single ? (t => getTrialScore(t)) : (t => getTrialRank(t));
    const scores = displayTrials.map(colorValue).filter(isFinite);
    const { min: sMin, max: sMax } = minMax(scores);
    const sRange = sMax - sMin || 1;

    // Draw axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.font = '10px Inter';
    ctx.fillStyle = '#8c8cbc';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) {
        const x = pad.left + i * gap;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + ph);
        ctx.stroke();
        ctx.fillText(names[i], x, h - 8);
        // Min/max labels (show choice labels for categorical)
        ctx.fillStyle = '#8c8cbc';
        if (ranges[i].categorical && ranges[i].choices) {
            const choices = ranges[i].choices;
            ctx.fillText(choices[choices.length - 1], x, pad.top - 6);
            ctx.fillText(choices[0], x, pad.top + ph + 14);
        } else {
            ctx.fillText(fmt(ranges[i].max), x, pad.top - 6);
            ctx.fillText(fmt(ranges[i].min), x, pad.top + ph + 14);
        }
        ctx.fillStyle = '#8c8cbc';
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
    for (const trial of displayTrials) {
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
function getTableColumns() {
    const columns = [
        { source: 'builtin', name: 'trial_id' },
        { source: 'builtin', name: 'rank' },
        ...S.paramNames.map(name => ({ source: 'param', name })),
        ...S.metricNames.map(name => ({ source: 'metric', name })),
    ];
    const nameCounts = new Map();
    for (const column of columns) {
        nameCounts.set(column.name, (nameCounts.get(column.name) || 0) + 1);
    }
    const sourceLabels = { builtin: 'trial field', param: 'parameter', metric: 'metric' };
    const sourcePrefixes = { builtin: 'trial', param: 'param', metric: 'metric' };
    const usedLabels = new Set();
    return columns.map(column => {
        const baseLabel = nameCounts.get(column.name) > 1
            ? `${sourcePrefixes[column.source]} · ${column.name}`
            : column.name;
        let label = baseLabel;
        for (let suffix = 2; usedLabels.has(label); suffix++) label = `${baseLabel} [${suffix}]`;
        usedLabels.add(label);
        return {
            ...column,
            key: `${column.source}:${column.name}`,
            label,
            accessibleLabel: `${sourceLabels[column.source]} ${column.name}`,
        };
    });
}

function reconcileTableSort() {
    if (!S.sortColumn) return;
    const replacement = getTableColumns()
        .find(column => column.key === S.sortColumn.key);
    if (replacement) {
        S.sortCol = replacement.name;
        S.sortColumn = replacement;
    } else {
        S.sortCol = null;
        S.sortColumn = null;
        S.sortAsc = true;
    }
}

function renderTable() {
    const thead = document.getElementById('trial-thead').querySelector('tr');
    const tbody = document.getElementById('trial-tbody');

    // Build columns
    const cols = getTableColumns();

    clearElement(thead);
    for (const [columnIndex, c] of cols.entries()) {
        const th = document.createElement('th');
        th.scope = 'col';
        if (S.sortColumn?.key === c.key) {
            th.className = S.sortAsc ? 'sorted-asc' : 'sorted-desc';
            th.setAttribute('aria-sort', S.sortAsc ? 'ascending' : 'descending');
        }
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = c.label;
        button.setAttribute('aria-label', `Sort trials by ${c.accessibleLabel}`);
        // Keep native button keyboard semantics without changing the compact
        // table-header appearance.
        button.style.background = 'transparent';
        button.style.border = '0';
        button.style.borderRadius = '0';
        button.style.color = 'inherit';
        button.style.display = 'block';
        button.style.font = 'inherit';
        button.style.padding = '0';
        button.style.textAlign = 'inherit';
        button.style.width = '100%';
        button.style.cursor = 'pointer';
        button.addEventListener('click', () => {
            const restoreFocus = document.activeElement === button;
            sortTable(c);
            if (restoreFocus) {
                const replacement = thead.children[columnIndex]?.querySelector('button');
                replacement?.focus({ preventScroll: true });
            }
        });
        th.appendChild(button);
        thead.appendChild(th);
    }

    const maxRows = 1000;
    let rows;
    if (S.sortColumn) {
        rows = getSortedTableEntries(maxRows).map(entry => entry.trial);
    } else {
        // The unsorted view is newest-first-at-the-bottom and only needs the
        // visible tail; avoid copying the other 99k+ rows.
        rows = S.trials.length > maxRows ? S.trials.slice(-maxRows) : S.trials;
    }
    if (S.trials.length > maxRows) {
        document.getElementById('table-count').textContent =
            `${S.trials.length} trials (showing ${maxRows})`;
    }

    clearElement(tbody);
    for (const t of rows) {
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

function getCellValue(trial, column) {
    if (!column) return NaN;
    if (column.source === 'builtin' && column.name === 'trial_id') return trial.trial_id;
    if (column.source === 'builtin' && column.name === 'rank') return trial.rank;
    if (column.source === 'param') return trial.params?.[column.name] ?? NaN;
    if (column.source === 'metric') return trial.metrics?.[column.name] ?? NaN;
    return NaN;
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
function compareCellValues(a, b, asc, column = null) {
    const na = normalizeSortValue(a);
    const nb = normalizeSortValue(b);
    const aMissing = na == null || (typeof na === 'number' && isNaN(na));
    const bMissing = nb == null || (typeof nb === 'number' && isNaN(nb));
    if (aMissing || bMissing) {
        if (aMissing && bMissing) return 0;
        return aMissing ? 1 : -1; // missing last in both directions
    }

    // Categorical parameters have an operator-declared order. Preserve that
    // order in the table instead of alphabetizing their display labels, which
    // would disagree with the stable indices used by parallel coordinates.
    const parameter = column == null ? null : S.space.find(space => space.name === column);
    if (parameter?.type === 'categorical' && Array.isArray(parameter.choices)) {
        const aIndex = parameter.choices.indexOf(na);
        const bIndex = parameter.choices.indexOf(nb);
        const aUnknown = aIndex < 0;
        const bUnknown = bIndex < 0;
        if (aUnknown || bUnknown) {
            if (aUnknown && bUnknown) return String(na).localeCompare(String(nb));
            return aUnknown ? 1 : -1; // undeclared values sort last in either direction
        }
        const cmp = aIndex - bIndex;
        return asc ? cmp : -cmp;
    }

    let cmp;
    if (typeof na === 'number' && typeof nb === 'number') {
        cmp = na < nb ? -1 : na > nb ? 1 : 0;
    } else {
        cmp = String(na).localeCompare(String(nb));
    }
    return asc ? cmp : -cmp;
}

function tableCacheKey() {
    return `${S.sortColumn?.key || ''}\u0000${S.sortAsc ? 'asc' : 'desc'}`;
}

function compareTableEntries(a, b) {
    const cmp = compareCellValues(
        getCellValue(a.trial, S.sortColumn),
        getCellValue(b.trial, S.sortColumn),
        S.sortAsc,
        S.sortColumn?.source === 'param' ? S.sortColumn.name : null,
    );
    return cmp || a.index - b.index;
}

function heapPushWorstFirst(heap, entry) {
    heap.push(entry);
    let idx = heap.length - 1;
    while (idx > 0) {
        const parent = Math.floor((idx - 1) / 2);
        if (compareTableEntries(heap[parent], heap[idx]) >= 0) break;
        [heap[parent], heap[idx]] = [heap[idx], heap[parent]];
        idx = parent;
    }
}

function heapRestoreWorstFirst(heap, idx = 0) {
    for (;;) {
        const left = idx * 2 + 1;
        const right = left + 1;
        let worst = idx;
        if (left < heap.length && compareTableEntries(heap[left], heap[worst]) > 0) worst = left;
        if (right < heap.length && compareTableEntries(heap[right], heap[worst]) > 0) worst = right;
        if (worst === idx) return;
        [heap[idx], heap[worst]] = [heap[worst], heap[idx]];
        idx = worst;
    }
}

// Select only the visible top-k rows. This preserves exact table ordering while
// bounding allocations to k entries instead of copying/sorting full history.
function getSortedTableEntries(limit) {
    const key = tableCacheKey();
    if (S.tableCache?.key === key && S.tableCache.sourceLength === S.trials.length) {
        return S.tableCache.entries;
    }

    const heap = [];
    for (let index = 0; index < S.trials.length; index++) {
        const entry = { trial: S.trials[index], index };
        if (heap.length < limit) {
            heapPushWorstFirst(heap, entry);
        } else if (compareTableEntries(entry, heap[0]) < 0) {
            heap[0] = entry;
            heapRestoreWorstFirst(heap);
        }
    }
    heap.sort(compareTableEntries);
    S.tableCache = { key, entries: heap, sourceLength: S.trials.length };
    return heap;
}

function updateTableCacheForAppend(trial, index) {
    const cache = S.tableCache;
    if (!cache || cache.key !== tableCacheKey() || cache.sourceLength !== index) {
        S.tableCache = null;
        return;
    }
    const entry = { trial, index };
    let low = 0, high = cache.entries.length;
    while (low < high) {
        const middle = (low + high) >>> 1;
        if (compareTableEntries(cache.entries[middle], entry) <= 0) low = middle + 1;
        else high = middle;
    }
    if (low < 1000) {
        cache.entries.splice(low, 0, entry);
        if (cache.entries.length > 1000) cache.entries.pop();
    }
    cache.sourceLength++;
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

function sortTable(column) {
    let resolved = column;
    if (typeof column === 'string') {
        const columns = getTableColumns();
        resolved = columns.find(candidate => candidate.key === column)
            || columns.find(candidate => candidate.name === column);
    }
    if (!resolved) return;
    if (S.sortColumn?.key !== resolved.key) {
        S.sortCol = resolved.name;
        S.sortColumn = resolved;
        S.sortAsc = true;
    } else if (S.sortAsc) {
        S.sortColumn = resolved;
        S.sortAsc = false;
    } else {
        S.sortCol = null;
        S.sortColumn = null;
        S.sortAsc = true;
    }
    S.tableCache = null;
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
        replaceTrials(await resp.json());
    }
    renderAll();
}

function previewObjectives() {
    S.previewActive = true;
    document.getElementById('preview-badge').style.display = '';
    for (const trial of S.trials) {
        rescoreTrialForPreview(trial);
        // Server ranks describe the previous objectives. Force the offline
        // ranker to rebuild them from the preview score vectors.
        delete trial.rank;
        delete trial.pareto_front;
    }
    computeRanksIfMissing(S.trials);
    S.scoreGroupCount = 0;
    for (const trial of S.trials) {
        S.scoreGroupCount = Math.max(S.scoreGroupCount, scoreGroupWidth(trial));
    }
    S.paretoCache = null;
    S.tableCache = null;
    rebuildBestAndConvergence();
    renderAll();
}

function rescoreTrialForPreview(trial) {
    const m = trial.metrics;
    if (!m || typeof m !== 'object') return;
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
        replaceTrials(await trialsResp.json());
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
        format: 'hola-dashboard-export',
        format_version: 1,
        exported_at: new Date().toISOString(),
        space: S.space,
        objectives: S.objectives,
        trials: S.trials,
    };
    const blob = new Blob([JSON.stringify(data, (_key, value) => {
        if (typeof value !== 'number' || Number.isFinite(value)) return value;
        if (Number.isNaN(value)) return 'nan';
        return value > 0 ? 'inf' : '-inf';
    }, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hola_trials_${Date.now()}.json`;
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
