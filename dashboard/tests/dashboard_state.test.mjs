// Copyright 2026 BlackRock, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { JSDOM } from 'jsdom';

const HERE = dirname(fileURLToPath(import.meta.url));
const DASHBOARD_DIR = resolve(HERE, '..');
const APP_JS = resolve(DASHBOARD_DIR, 'app.js');
const INDEX_HTML = resolve(DASHBOARD_DIR, 'index.html');

function buildHtml() {
    const appSrc = readFileSync(APP_JS, 'utf8');
    const bridge = `
        window.S = S;
        for (const name of [
            'applyCheckpointData',
            'checkpointDataToDashboard',
            'computeRanksIfMissing',
            'connectToServer',
            'handleEngineEvent',
            'isTrialFeasible',
            'isCheckpointFileSizeAllowed',
            'loadCheckpointFile',
            'previewObjectives',
            'renderConvergence',
            'renderParallel',
            'renderPareto',
            'renderTable',
            'replaceTrials',
            'sortTable',
            'resyncLiveState',
            'stopStream',
            'upsertTrial',
        ]) window[name] = eval(name);
        window.MAX_CHECKPOINT_BYTES = MAX_CHECKPOINT_BYTES;
    `;
    return readFileSync(INDEX_HTML, 'utf8').replace(
        /<script\b[^>]*\bsrc=["']app\.js["']><\/script>/i,
        `<script>${appSrc.replace(/<\/script>/gi, '<\\/script>')}</script>\n`
        + `<script>${bridge}</script>`,
    );
}

function createDom(fetchImpl = () => new Promise(() => {})) {
    const alerts = [];
    const canvasCalls = [];
    const dom = new JSDOM(buildHtml(), {
        url: pathToFileURL(INDEX_HTML).href,
        runScripts: 'dangerously',
        pretendToBeVisual: true,
        beforeParse(window) {
            window.fetch = fetchImpl;
            window.Headers = globalThis.Headers;
            window.TextDecoder = globalThis.TextDecoder;
            window.requestAnimationFrame = () => 0;
            window.cancelAnimationFrame = () => {};
            window.alert = message => alerts.push(String(message));
            window.confirm = () => true;
        },
    });
    const context = {
        beginPath() {},
        clearRect() {},
        fill() {},
        fillText(text, x, y) { canvasCalls.push({ op: 'fillText', text, x, y }); },
        lineTo(x, y) { canvasCalls.push({ op: 'lineTo', x, y }); },
        moveTo(x, y) { canvasCalls.push({ op: 'moveTo', x, y }); },
        restore() {},
        rotate() {},
        save() {},
        scale() {},
        setLineDash() {},
        stroke() {},
        translate() {},
        arc() {},
    };
    for (const canvas of dom.window.document.querySelectorAll('canvas')) {
        canvas.getContext = () => context;
    }
    return { dom, window: dom.window, alerts, canvasCalls };
}

function deferred() {
    let resolvePromise;
    let rejectPromise;
    const promise = new Promise((resolve, reject) => {
        resolvePromise = resolve;
        rejectPromise = reject;
    });
    return { promise, resolve: resolvePromise, reject: rejectPromise };
}

function jsonResponse(value, status = 200) {
    return {
        ok: status >= 200 && status < 300,
        status,
        body: null,
        json: async () => value,
    };
}

function requestRouter() {
    const requests = new Map();
    const fetch = (url, options = {}) => {
        const key = String(url);
        const request = deferred();
        request.options = options;
        const queue = requests.get(key) || [];
        queue.push(request);
        requests.set(key, queue);
        return request.promise;
    };
    const waitFor = async (url, occurrence = 0) => {
        for (let i = 0; i < 100; i++) {
            const request = requests.get(url)?.[occurrence];
            if (request) return request;
            await new Promise(resolvePromise => setImmediate(resolvePromise));
        }
        throw new Error(`request was not issued: ${url} (occurrence ${occurrence})`);
    };
    return { fetch, requests, waitFor };
}

test('full checkpoints retain embedded categorical space and objectives', () => {
    const { dom, window } = createDom();
    try {
        window.applyCheckpointData({
            config: {
                space: {
                    depth: { type: 'integer', min: 1, max: 8 },
                    optimizer: { type: 'categorical', choices: ['adam', 'sgd'] },
                },
                objectives: [
                    { field: 'loss', obj_type: 'minimize', priority: 1, group: 'quality' },
                    { field: 'latency', obj_type: 'minimize', priority: 2, group: 'cost' },
                ],
            },
            checkpoint: {
                leaderboard: {
                    trials: [{
                        trial_id: 7,
                        candidate: { depth: 4, optimizer: 'adam' },
                        raw_metrics: { loss: 0.2, latency: 5 },
                        observation: { quality: 0.2, cost: 10 },
                    }],
                },
            },
        });

        const categorical = window.S.space.find(param => param.name === 'optimizer');
        assert.ok(categorical, 'categorical parameter should come from embedded config');
        assert.equal(categorical.type, 'categorical');
        assert.deepEqual(Array.from(categorical.choices), ['adam', 'sgd']);
        assert.deepEqual(Array.from(window.S.paramNames), ['depth', 'optimizer']);
        assert.equal(window.S.objectives.length, 2);
        assert.equal(window.S.objectives[1].field, 'latency');
        assert.equal(window.S.objectives[1].group, 'cost');
        assert.equal(window.S.serverObjectives[0].field, 'loss');
        assert.equal(window.S.mode, 'offline');
    } finally {
        dom.window.close();
    }
});

test('mixed-space rendering and sorting preserve declared categorical order', () => {
    const { dom, window, canvasCalls } = createDom();
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            space: [
                { name: 'learning_rate', type: 'real', min: 0.001, max: 0.1, scale: 'log10' },
                { name: 'layers', type: 'integer', min: 1, max: 8, scale: 'linear' },
                // Deliberately non-lexical: declared order must win.
                { name: 'optimizer', type: 'categorical', choices: ['zeta', 'alpha'] },
            ],
            objectives: [{ field: 'loss', obj_type: 'minimize' }],
            trials: [
                {
                    trial_id: 0,
                    params: { learning_rate: 0.01, layers: 2, optimizer: 'alpha' },
                    metrics: { loss: 0.4 },
                    score_vector: { loss: 0.4 },
                },
                {
                    trial_id: 1,
                    params: { learning_rate: 0.02, layers: 4, optimizer: 'zeta' },
                    metrics: { loss: 0.2 },
                    score_vector: { loss: 0.2 },
                },
            ],
        });

        const canvas = window.document.getElementById('parallel-canvas');
        Object.defineProperty(canvas.parentElement, 'clientWidth', {
            configurable: true,
            value: 640,
        });
        window.renderParallel();
        const labels = canvasCalls
            .filter(call => call.op === 'fillText')
            .map(call => String(call.text));
        assert.ok(labels.includes('zeta'));
        assert.ok(labels.includes('alpha'));
        assert.ok(
            canvasCalls
                .filter(call => call.op === 'lineTo' || call.op === 'moveTo')
                .every(call => Number.isFinite(call.x) && Number.isFinite(call.y)),
            'mixed-space canvas coordinates must remain finite',
        );

        const optimizerColumn = [...window.document.querySelectorAll('#trial-thead th')]
            .findIndex(header => header.textContent === 'optimizer');
        assert.ok(optimizerColumn >= 0);

        const optimizerHeader = () => [...window.document.querySelectorAll('#trial-thead th')]
            .find(header => header.textContent === 'optimizer');
        assert.equal(optimizerHeader().getAttribute('role'), null);
        assert.equal(optimizerHeader().scope, 'col');
        assert.equal(optimizerHeader().querySelector('button')?.type, 'button');
        assert.equal(optimizerHeader().querySelector('button')?.tabIndex, 0);

        optimizerHeader().querySelector('button').click();
        let rows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(rows[0].children[optimizerColumn].textContent, 'zeta');
        assert.equal(rows[1].children[optimizerColumn].textContent, 'alpha');
        assert.equal(optimizerHeader().getAttribute('aria-sort'), 'ascending');
        assert.equal(window.document.querySelectorAll('#trial-thead th[aria-sort]').length, 1);

        optimizerHeader().querySelector('button').click();
        rows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(rows[0].children[optimizerColumn].textContent, 'alpha');
        assert.equal(rows[1].children[optimizerColumn].textContent, 'zeta');
        assert.equal(optimizerHeader().getAttribute('aria-sort'), 'descending');

        optimizerHeader().querySelector('button').click();
        rows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(window.S.sortCol, null);
        assert.equal(optimizerHeader().getAttribute('aria-sort'), null);
        assert.equal(window.document.querySelectorAll('#trial-thead th[aria-sort]').length, 0);
        assert.equal(rows[0].children[optimizerColumn].textContent, 'alpha');
        assert.equal(rows[1].children[optimizerColumn].textContent, 'zeta');
    } finally {
        dom.window.close();
    }
});

test('live appends replace scalar best and demote dominated Pareto-front trials', () => {
    const { dom, window } = createDom();
    try {
        const xSelect = window.document.getElementById('pareto-x');
        const ySelect = window.document.getElementById('pareto-y');
        const selectAxes = (x, y) => {
            xSelect.replaceChildren(new window.Option(x, x));
            ySelect.replaceChildren(new window.Option(y, y));
            xSelect.value = x;
            ySelect.value = y;
        };

        window.S.objectives = [{ field: 'loss', obj_type: 'minimize' }];
        window.replaceTrials([{
            trial_id: 0,
            metrics: { loss: 10, latency: 1 },
            score_vector: { loss: 10 },
            rank: 0,
            pareto_front: 0,
        }]);
        selectAxes('loss', 'latency');
        window.renderPareto();
        const scalarCache = window.S.paretoCache;

        window.upsertTrial({
            trial_id: 1,
            metrics: { loss: 1, latency: 2 },
            score_vector: { loss: 1 },
            // Both completion DTOs legitimately arrived as rank zero at their
            // respective commit times; the client must demote the stale best.
            rank: 0,
            pareto_front: 0,
        });
        assert.equal(window.S.trials[window.S.bestIdx].trial_id, 1);
        assert.equal(window.S.trials[0].pareto_front, 1);
        assert.deepEqual(Array.from(window.S.paretoFrontIds), [1]);
        assert.deepEqual(Array.from(window.S.convergenceBest), [10, 1]);
        assert.strictEqual(window.S.paretoCache, scalarCache);
        assert.deepEqual(Array.from(
            window.S.paretoCache.front,
            point => point.trial.trial_id,
        ), [1]);

        window.S.objectives = [
            { field: 'quality', obj_type: 'minimize' },
            { field: 'cost', obj_type: 'minimize' },
        ];
        window.replaceTrials([
            {
                trial_id: 10,
                metrics: { quality: 2, cost: 2 },
                score_vector: { quality: 2, cost: 2 },
                rank: 0,
                pareto_front: 0,
            },
            {
                trial_id: 11,
                metrics: { quality: 0, cost: 3 },
                score_vector: { quality: 0, cost: 3 },
                rank: 1,
                pareto_front: 0,
            },
        ]);
        selectAxes('quality', 'cost');
        window.renderPareto();
        const vectorCache = window.S.paretoCache;

        window.upsertTrial({
            trial_id: 12,
            metrics: { quality: 1, cost: 1 },
            score_vector: { quality: 1, cost: 1 },
            rank: 1,
            pareto_front: 0,
        });
        assert.equal(window.S.trials[0].pareto_front, 1);
        assert.deepEqual(Array.from(window.S.paretoFrontIds), [11, 12]);
        assert.equal(window.S.trials[window.S.bestIdx].trial_id, 11);
        assert.deepEqual(Array.from(window.S.convergenceBest), [1, 2, 2]);
        assert.strictEqual(window.S.paretoCache, vectorCache);
        assert.deepEqual(
            Array.from(
                window.S.paretoCache.front,
                point => point.trial.trial_id,
            ).sort((a, b) => a - b),
            [11, 12],
        );
        assert.ok(window.S.paretoCache.dominated.some(point => point.trial.trial_id === 10));
    } finally {
        dom.window.close();
    }
});

test('100k checkpoint state load and incremental update burst keep caches bounded', {
    timeout: 30_000,
}, () => {
    const { dom, window } = createDom();
    try {
        const trialCount = 100_000;
        const trials = Array.from({ length: trialCount }, (_, i) => ({
            trial_id: i,
            params: { x: i % 1000 },
            metrics: { loss: trialCount - i, latency: i % 1000 },
            score_vector: { loss: trialCount - i },
            rank: i + 1,
            pareto_front: i + 1,
        }));
        const checkpoint = {
            format: 'hola-dashboard-export',
            space: [{ name: 'x', type: 'integer', min: 0, max: 999 }],
            objectives: [{ field: 'loss', obj_type: 'minimize' }],
            trials,
        };
        const checkpointBytes = new TextEncoder().encode(JSON.stringify(checkpoint)).byteLength;
        assert.ok(checkpointBytes > 8 * 1024 * 1024, 'fixture must exceed the former 8 MiB cap');
        assert.equal(window.isCheckpointFileSizeAllowed(checkpointBytes), true);
        assert.equal(window.isCheckpointFileSizeAllowed(window.MAX_CHECKPOINT_BYTES), true);
        assert.equal(window.isCheckpointFileSizeAllowed(window.MAX_CHECKPOINT_BYTES + 1), false);

        const loadStarted = performance.now();
        window.applyCheckpointData(checkpoint);
        const loadElapsed = performance.now() - loadStarted;
        assert.ok(loadElapsed < 10_000, `100k load took ${loadElapsed.toFixed(0)} ms`);
        assert.equal(window.S.trials.length, trialCount);
        assert.equal(window.S.trialIndex.size, trialCount);
        assert.equal(window.S.trialIndex.get(99_999), 99_999);
        assert.deepEqual(Array.from(window.S.metricNames), ['latency', 'loss']);
        assert.equal(window.S.convergenceScores.length, trialCount);
        assert.equal(window.S.convergenceBest.at(-1), 1);

        const xSelect = window.document.getElementById('pareto-x');
        const ySelect = window.document.getElementById('pareto-y');
        xSelect.add(new window.Option('loss', 'loss'));
        ySelect.add(new window.Option('latency', 'latency'));
        xSelect.value = 'loss';
        ySelect.value = 'latency';
        window.renderPareto();
        const paretoCache = window.S.paretoCache;
        assert.equal(paretoCache.sourceLength, trialCount);
        assert.ok(paretoCache.front.length + paretoCache.dominated.length <= 5000);

        // Building a sorted view retains only the 1,000 visible rows. The
        // cache is then maintained incrementally across the append burst.
        window.sortTable('trial_id');
        assert.equal(window.S.tableCache.entries.length, 1000);
        assert.equal(window.S.tableCache.entries[0].trial.trial_id, 0);
        assert.equal(window.S.tableCache.entries.at(-1).trial.trial_id, 999);
        const convergenceScores = window.S.convergenceScores;
        const convergenceBest = window.S.convergenceBest;

        const burstCount = 2000;
        const burstStarted = performance.now();
        for (let offset = 0; offset < burstCount; offset++) {
            const id = trialCount + offset;
            window.upsertTrial({
                trial_id: id,
                params: { x: id % 1000 },
                metrics: { loss: -(offset + 1), latency: offset, throughput: offset * 2 },
                score_vector: { loss: -(offset + 1) },
                rank: offset === burstCount - 1 ? 0 : id + 1,
                pareto_front: offset === burstCount - 1 ? 0 : id + 1,
            });
        }
        const burstElapsed = performance.now() - burstStarted;
        assert.ok(burstElapsed < 5000, `2k state-update burst took ${burstElapsed.toFixed(0)} ms`);

        assert.equal(window.S.trials.length, trialCount + burstCount);
        assert.equal(window.S.trialIndex.get(trialCount + burstCount - 1), trialCount + burstCount - 1);
        assert.deepEqual(Array.from(window.S.metricNames), ['latency', 'loss', 'throughput']);
        assert.strictEqual(window.S.convergenceScores, convergenceScores);
        assert.strictEqual(window.S.convergenceBest, convergenceBest);
        assert.equal(window.S.convergenceScores.length, trialCount + burstCount);
        assert.equal(window.S.convergenceBest.at(-1), -burstCount);
        assert.equal(window.S.trials[window.S.bestIdx].trial_id, trialCount + burstCount - 1);
        assert.strictEqual(window.S.paretoCache, paretoCache);
        assert.equal(window.S.paretoCache.sourceLength, trialCount + burstCount);
        assert.equal(window.S.tableCache.sourceLength, trialCount + burstCount);
        assert.equal(window.S.tableCache.entries.length, 1000);

        const replay = JSON.parse(JSON.stringify(window.S.trials.at(-1)));
        assert.equal(window.upsertTrial(replay), false);
        assert.strictEqual(window.S.convergenceScores, convergenceScores);
        assert.strictEqual(window.S.paretoCache, paretoCache);
        assert.equal(window.S.trials.length, trialCount + burstCount);
    } finally {
        dom.window.close();
    }
});

test('large unranked vector checkpoints use exact 2-D and bounded many-objective ranking', {
    timeout: 15_000,
}, () => {
    const { dom, window } = createDom();
    try {
        const trialCount = 100_000;
        const trials = Array.from({ length: trialCount }, (_, i) => ({
            trial_id: i,
            score_vector: {
                first: i,
                second: trialCount - i,
            },
        }));

        const exactStarted = performance.now();
        window.computeRanksIfMissing(trials);
        const exactElapsed = performance.now() - exactStarted;
        assert.ok(exactElapsed < 5000, `100k two-objective ranking took ${exactElapsed.toFixed(0)} ms`);
        assert.equal(trials[0].pareto_front, 0);
        assert.equal(trials[Math.floor(trialCount / 2)].pareto_front, 0);
        assert.equal(trials.at(-1).pareto_front, 0);

        for (let i = 0; i < trials.length; i++) {
            trials[i].score_vector.third = i % 17;
            delete trials[i].rank;
            delete trials[i].pareto_front;
        }
        trials.push({
            trial_id: trialCount,
            score_vector: { first: 'inf', second: 0, third: 0 },
            rank: 0,
            pareto_front: 0,
        });

        const started = performance.now();
        window.computeRanksIfMissing(trials);
        const elapsed = performance.now() - started;
        assert.ok(elapsed < 5000, `100k many-objective ranking took ${elapsed.toFixed(0)} ms`);

        const frontZero = trials.filter(trial => trial.pareto_front === 0);
        assert.deepEqual(frontZero.map(trial => trial.trial_id), [0]);
        assert.ok(frontZero.every(window.isTrialFeasible));
        assert.ok(trials.at(-1).pareto_front > 0);
        assert.ok(trials.at(-1).rank > 0);
    } finally {
        dom.window.close();
    }
});

test('non-finite vector trials are never assigned Pareto front zero', () => {
    const { dom, window } = createDom();
    try {
        const trials = [
            { trial_id: 0, score_vector: { quality: 1, cost: 3 } },
            { trial_id: 1, score_vector: { quality: 3, cost: 1 } },
            { trial_id: 2, score_vector: { quality: 4, cost: 4 } },
            { trial_id: 5, score_vector: { quality: 5, cost: 5 } },
            // Simulate an older dashboard export that persisted the buggy rank.
            {
                trial_id: 3,
                score_vector: { quality: Infinity, cost: Infinity },
                rank: 0,
                pareto_front: 0,
            },
            { trial_id: 4, score_vector: { quality: null, cost: 2 } },
        ];

        window.computeRanksIfMissing(trials);

        assert.equal(trials[0].pareto_front, 0);
        assert.equal(trials[1].pareto_front, 0);
        assert.equal(trials[2].pareto_front, 1);
        assert.equal(trials[3].pareto_front, 2);
        assert.ok(trials[4].pareto_front > 0);
        assert.ok(trials[4].rank > 0);
        assert.ok(trials[5].pareto_front > 0);
        assert.ok(trials[5].rank > 0);
        const worstFeasible = Math.max(...trials.slice(0, 4).map(trial => trial.rank));
        assert.ok(trials[4].rank > worstFeasible);
        assert.ok(trials[5].rank > worstFeasible);
    } finally {
        dom.window.close();
    }
});

test('live infeasible trials are never highlighted as Pareto-optimal', () => {
    const { dom, window } = createDom();
    try {
        // Score vectors are already oriented by the server: maximize accuracy
        // becomes a negative cost. Trials 0/1 trade off, trial 2 is dominated,
        // and trial 3 violates the latency constraint despite excellent raw
        // metrics. Simulate an older/stale server incorrectly labelling it 0.
        window.replaceTrials([
            {
                trial_id: 0,
                metrics: { loss: 1, accuracy: 0.8, latency: 5 },
                score_vector: { loss: 1, accuracy: -0.8, latency: 0.5 },
                rank: 0,
                pareto_front: 0,
            },
            {
                trial_id: 1,
                metrics: { loss: 2, accuracy: 0.9, latency: 5 },
                score_vector: { loss: 2, accuracy: -0.9, latency: 0.5 },
                rank: 1,
                pareto_front: 0,
            },
            {
                trial_id: 2,
                metrics: { loss: 2, accuracy: 0.7, latency: 5 },
                score_vector: { loss: 2, accuracy: -0.7, latency: 0.5 },
                rank: 2,
                pareto_front: 1,
            },
            {
                trial_id: 3,
                metrics: { loss: 0.5, accuracy: 0.95, latency: 20 },
                score_vector: { loss: 0.5, accuracy: -0.95, latency: 'inf' },
                rank: 0,
                pareto_front: 0,
            },
        ]);
        window.S.objectives = [
            { field: 'loss', obj_type: 'minimize' },
            { field: 'accuracy', obj_type: 'maximize' },
            { field: 'latency', obj_type: 'minimize', target: 0, limit: 10 },
        ];
        const xSelect = window.document.getElementById('pareto-x');
        const ySelect = window.document.getElementById('pareto-y');
        xSelect.add(new window.Option('loss', 'loss'));
        ySelect.add(new window.Option('accuracy', 'accuracy'));
        xSelect.value = 'loss';
        ySelect.value = 'accuracy';
        const canvas = window.document.getElementById('pareto-canvas');
        Object.defineProperty(canvas.parentElement, 'clientWidth', {
            configurable: true,
            value: 640,
        });

        window.renderPareto();

        const points = new Map(window.S._paretoPoints.map(point => [point.trial.trial_id, point]));
        assert.equal(points.get(0).onFront, true);
        assert.equal(points.get(1).onFront, true);
        assert.equal(points.get(2).onFront, false);
        assert.equal(points.get(3).onFront, false);
        assert.equal(window.isTrialFeasible(window.S.trials[3]), false);
    } finally {
        dom.window.close();
    }
});

test('objective preview recomputes best rank and front from preview scores', () => {
    const { dom, window } = createDom();
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            space: [],
            objectives: [{ field: 'loss', obj_type: 'minimize', priority: 1 }],
            trials: [
                {
                    trial_id: 0,
                    metrics: { loss: 1 },
                    score_vector: { loss: 1 },
                    rank: 0,
                    pareto_front: 0,
                },
                {
                    trial_id: 1,
                    metrics: { loss: 2 },
                    score_vector: { loss: 2 },
                    rank: 1,
                    pareto_front: 1,
                },
            ],
        });
        window.S.objectives = [
            { field: 'loss', obj_type: 'maximize', priority: 1 },
        ];

        window.previewObjectives();

        assert.equal(window.S.trials[0].score_vector.loss, -1);
        assert.equal(window.S.trials[1].score_vector.loss, -2);
        assert.equal(window.S.trials[0].rank, 1);
        assert.equal(window.S.trials[0].pareto_front, 1);
        assert.equal(window.S.trials[1].rank, 0);
        assert.equal(window.S.trials[1].pareto_front, 0);
        assert.equal(window.S.trials[window.S.bestIdx].trial_id, 1);
        assert.deepEqual(Array.from(window.S.paretoFrontIds), [1]);
    } finally {
        dom.window.close();
    }
});

test('stale connect and resync responses cannot overwrite a newer connection', async () => {
    const router = requestRouter();
    const { dom, window, alerts } = createDom(router.fetch);
    const input = window.document.getElementById('server-url');
    const server = 'http://server.test';
    const c = 'http://server-c.test';
    const trialsUrl = url => `${url}/api/trials?sorted_by=index&include_infeasible=true`;

    try {
        // Use the same URL for both attempts: a URL-only stale-response check
        // cannot distinguish these, so this specifically exercises generation.
        input.value = server;
        const staleConnect = window.connectToServer();
        const staleInitialCursor = await router.waitFor(`${server}/api/event_cursor`, 0);

        input.value = server;
        const freshConnect = window.connectToServer();
        const freshCursor = await router.waitFor(`${server}/api/event_cursor`, 1);
        freshCursor.resolve(jsonResponse({ last_event_id: '20' }));
        const freshSpace = await router.waitFor(`${server}/api/space`, 0);
        freshSpace.resolve(jsonResponse({ params: [{ name: 'fresh', type: 'real' }] }));
        const freshObjectives = await router.waitFor(`${server}/api/objectives`);
        freshObjectives.resolve(jsonResponse({ objectives: [{ field: 'loss', obj_type: 'minimize' }] }));
        const freshTrials = await router.waitFor(trialsUrl(server));
        freshTrials.resolve(jsonResponse([{ trial_id: 20, score_vector: { loss: 1 }, rank: 0 }]));
        await freshConnect;

        assert.equal(window.S.serverUrl, server);
        assert.equal(window.S.space[0].name, 'fresh');
        assert.equal(window.S.trials[0].trial_id, 20);

        // The first connect resolves last and must stop before issuing any more
        // requests or committing its stale data, despite using the same URL.
        staleInitialCursor.resolve(jsonResponse({ last_event_id: '10' }));
        await staleConnect;
        assert.equal(router.requests.get(`${server}/api/space`).length, 1);
        assert.equal(window.S.space[0].name, 'fresh');

        // Begin a full resync, then supersede it with connection C.
        const staleResync = window.resyncLiveState();
        const staleCursor = await router.waitFor(`${server}/api/event_cursor`, 2);
        staleCursor.resolve(jsonResponse({ last_event_id: '21' }));
        const staleSpace = await router.waitFor(`${server}/api/space`, 1);
        const staleObjectives = await router.waitFor(`${server}/api/objectives`, 1);
        const staleTrials = await router.waitFor(trialsUrl(server), 1);

        input.value = c;
        const connectC = window.connectToServer();
        const cCursor = await router.waitFor(`${c}/api/event_cursor`);
        cCursor.resolve(jsonResponse({ last_event_id: '30' }));
        const cSpace = await router.waitFor(`${c}/api/space`);
        cSpace.resolve(jsonResponse({ params: [{ name: 'from_c', type: 'categorical', choices: ['x', 'y'] }] }));
        const cObjectives = await router.waitFor(`${c}/api/objectives`);
        cObjectives.resolve(jsonResponse({ objectives: [{ field: 'c_loss', obj_type: 'minimize' }] }));
        const cTrials = await router.waitFor(trialsUrl(c));
        cTrials.resolve(jsonResponse([{ trial_id: 30, score_vector: { c_loss: 2 }, rank: 0 }]));
        await connectC;

        staleSpace.resolve(jsonResponse({ params: [{ name: 'stale_resync', type: 'real' }] }));
        staleObjectives.resolve(jsonResponse({ objectives: [{ field: 'stale', obj_type: 'minimize' }] }));
        staleTrials.resolve(jsonResponse([{ trial_id: 999, rank: 0 }]));
        await staleResync;

        assert.equal(window.S.serverUrl, c);
        assert.equal(window.S.space[0].name, 'from_c');
        assert.equal(window.S.objectives[0].field, 'c_loss');
        assert.equal(window.S.trials[0].trial_id, 30);
        assert.deepEqual(alerts, []);
    } finally {
        window.stopStream();
        dom.window.close();
    }
});

test('ObjectivesChanged event performs a cursor-safe full live resync', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch);
    const server = 'http://objectives-changed.test';
    const trialsUrl = `${server}/api/trials?sorted_by=index&include_infeasible=true`;
    try {
        window.S.connectionGeneration = 7;
        window.S.serverUrl = server;
        window.S.mode = 'live';
        window.S.trials = [{ trial_id: 0, score_vector: { old: 9 }, rank: 0 }];

        const handling = window.handleEngineEvent({ type: 'ObjectivesChanged' }, 7, server);
        (await router.waitFor(`${server}/api/event_cursor`))
            .resolve(jsonResponse({ last_event_id: '44' }));
        (await router.waitFor(`${server}/api/space`))
            .resolve(jsonResponse({ params: [{ name: 'x', type: 'real', min: 0, max: 1 }] }));
        (await router.waitFor(`${server}/api/objectives`))
            .resolve(jsonResponse({ objectives: [{ field: 'new_loss', obj_type: 'minimize' }] }));
        (await router.waitFor(trialsUrl)).resolve(jsonResponse([{
            trial_id: 0,
            metrics: { new_loss: 1 },
            score_vector: { new_loss: 1 },
            rank: 0,
            pareto_front: 0,
        }]));
        await handling;

        assert.equal(window.S.lastEventId, '44');
        assert.equal(window.S.objectives[0].field, 'new_loss');
        assert.equal(window.S.trials[0].score_vector.new_loss, 1);
        const stream = await router.waitFor(`${server}/api/events`);
        assert.equal(stream.options.headers.get('Last-Event-ID'), '44');
    } finally {
        window.stopStream();
        dom.window.close();
    }
});

test('failed resync retries and replays an event racing its replacement snapshot', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch);
    const server = 'http://resync-race.test';
    const trialsUrl = `${server}/api/trials?sorted_by=index&include_infeasible=true`;
    const input = window.document.getElementById('server-url');
    try {
        input.value = server;
        const connecting = window.connectToServer();
        (await router.waitFor(`${server}/api/event_cursor`, 0))
            .resolve(jsonResponse({ last_event_id: '0' }));
        (await router.waitFor(`${server}/api/space`, 0))
            .resolve(jsonResponse({ params: [] }));
        (await router.waitFor(`${server}/api/objectives`, 0))
            .resolve(jsonResponse({ objectives: [{ field: 'loss', obj_type: 'minimize' }] }));
        (await router.waitFor(trialsUrl, 0)).resolve(jsonResponse([{
            trial_id: 0,
            metrics: { loss: 10 },
            score_vector: { loss: 10 },
            rank: 0,
            pareto_front: 0,
        }]));
        await connecting;
        await router.waitFor(`${server}/api/events`, 0);

        // Make the production retry delay immediate and deterministic here.
        window.setTimeout = callback => {
            queueMicrotask(callback);
            return 1;
        };
        window.clearTimeout = () => {};
        const failedResync = window.resyncLiveState();
        (await router.waitFor(`${server}/api/event_cursor`, 1))
            .resolve(jsonResponse({}, 503));
        await failedResync;

        // Retry captures cursor 5 before taking a snapshot that does not yet
        // contain completion 6.
        (await router.waitFor(`${server}/api/event_cursor`, 2))
            .resolve(jsonResponse({ last_event_id: '5' }));
        (await router.waitFor(`${server}/api/space`, 1))
            .resolve(jsonResponse({ params: [] }));
        (await router.waitFor(`${server}/api/objectives`, 1))
            .resolve(jsonResponse({ objectives: [{ field: 'loss', obj_type: 'minimize' }] }));
        (await router.waitFor(trialsUrl, 1)).resolve(jsonResponse([{
            trial_id: 0,
            metrics: { loss: 10 },
            score_vector: { loss: 10 },
            rank: 0,
            pareto_front: 0,
        }]));

        const replayStream = await router.waitFor(`${server}/api/events`, 1);
        assert.equal(replayStream.options.headers.get('Last-Event-ID'), '5');
        const completion = {
            type: 'TrialCompleted',
            trial_id: 1,
            trial: {
                trial_id: 1,
                metrics: { loss: 1 },
                score_vector: { loss: 1 },
                rank: 0,
                pareto_front: 0,
            },
        };
        const encoded = new TextEncoder().encode(
            `id: 6\ndata: ${JSON.stringify(completion)}\n\n`,
        );
        let delivered = false;
        replayStream.resolve({
            ok: true,
            status: 200,
            body: {
                getReader() {
                    return {
                        read() {
                            if (!delivered) {
                                delivered = true;
                                return Promise.resolve({ value: encoded, done: false });
                            }
                            return new Promise(() => {});
                        },
                    };
                },
            },
        });

        for (let i = 0; i < 20 && window.S.trials.length < 2; i++) {
            await new Promise(resolvePromise => setImmediate(resolvePromise));
        }
        assert.deepEqual(window.S.trials.map(trial => trial.trial_id), [0, 1]);
        assert.equal(window.S.lastEventId, '6');
        assert.equal(window.S.trials[window.S.bestIdx].trial_id, 1);
    } finally {
        window.stopStream();
        dom.window.close();
    }
});

test('event watermark replays a completion racing the initial REST snapshot', async () => {
    const router = requestRouter();
    const { dom, window, alerts } = createDom(router.fetch);
    const server = 'http://snapshot-race.test';
    const trialsUrl = `${server}/api/trials?sorted_by=index&include_infeasible=true`;
    const input = window.document.getElementById('server-url');

    try {
        input.value = server;
        const connecting = window.connectToServer();
        (await router.waitFor(`${server}/api/event_cursor`))
            .resolve(jsonResponse({ last_event_id: '0' }));
        (await router.waitFor(`${server}/api/space`))
            .resolve(jsonResponse({ params: [{ name: 'x', type: 'real', min: 0, max: 1 }] }));
        (await router.waitFor(`${server}/api/objectives`))
            .resolve(jsonResponse({ objectives: [{ field: 'loss', obj_type: 'minimize' }] }));
        // Snapshot does not contain the racing completion.
        (await router.waitFor(trialsUrl)).resolve(jsonResponse([]));
        await connecting;

        const streamRequest = await router.waitFor(`${server}/api/events`);
        assert.equal(streamRequest.options.headers.get('Last-Event-ID'), '0');
        const event = {
            type: 'TrialCompleted',
            trial_id: 41,
            trial: {
                trial_id: 41,
                params: { x: 0.4 },
                metrics: { loss: 0.25 },
                score_vector: { loss: 0.25 },
                rank: 0,
                pareto_front: 0,
            },
        };
        const encoded = new TextEncoder().encode(`id: 1\ndata: ${JSON.stringify(event)}\n\n`);
        let firstRead = true;
        streamRequest.resolve({
            ok: true,
            status: 200,
            body: {
                getReader() {
                    return {
                        read() {
                            if (firstRead) {
                                firstRead = false;
                                return Promise.resolve({ value: encoded, done: false });
                            }
                            return new Promise(() => {});
                        },
                    };
                },
            },
        });

        for (let i = 0; i < 20 && window.S.trials.length === 0; i++) {
            await new Promise(resolvePromise => setImmediate(resolvePromise));
        }
        assert.equal(window.S.trials.length, 1);
        assert.equal(window.S.trials[0].trial_id, 41);
        assert.equal(window.S.lastEventId, '1');
        assert.deepEqual(alerts, []);
    } finally {
        window.stopStream();
        dom.window.close();
    }
});
