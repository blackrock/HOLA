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
const STYLES_CSS = resolve(DASHBOARD_DIR, 'styles.css');

function buildHtml() {
    const appSrc = readFileSync(APP_JS, 'utf8');
    const bridge = `
        window.S = S;
        for (const name of [
            'apiFetch',
            'apiJson',
            'applyCheckpointData',
            'checkpointDataToDashboard',
            'clearApiToken',
            'computeRanksIfMissing',
            'connectToServer',
            'handleEngineEvent',
            'isTrialFeasible',
            'isCheckpointFileSizeAllowed',
            'loadCheckpointFile',
            'previewObjectives',
            'rescoreTrialForPreview',
            'resetObjectives',
            'applyObjectives',
            'renderConvergence',
            'renderParallel',
            'renderPareto',
            'renderTable',
            'replaceTrials',
            'sortTable',
            'resyncLiveState',
            'startStream',
            'stopStream',
            'upsertTrial',
        ]) window[name] = eval(name);
        window.MAX_CHECKPOINT_BYTES = MAX_CHECKPOINT_BYTES;
        window.MAX_SSE_RECORD_CHARS = MAX_SSE_RECORD_CHARS;
        window.MAX_SSE_RECORDS_PER_READ = MAX_SSE_RECORDS_PER_READ;
        window.MAX_EVENT_FETCHES = MAX_EVENT_FETCHES;
    `;
    return readFileSync(INDEX_HTML, 'utf8').replace(
        /<script\b[^>]*\bsrc=["']app\.js["']><\/script>/i,
        `<script>${appSrc.replace(/<\/script>/gi, '<\\/script>')}</script>\n`
        + `<script>${bridge}</script>`,
    );
}

function createDom(fetchImpl = () => new Promise(() => {}), url = pathToFileURL(INDEX_HTML).href) {
    const alerts = [];
    const canvasCalls = [];
    const dom = new JSDOM(buildHtml(), {
        url,
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
        fillText(text, x, y) {
            canvasCalls.push({ op: 'fillText', text, x, y, fillStyle: this.fillStyle });
        },
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

        let optimizerButton = optimizerHeader().querySelector('button');
        optimizerButton.focus();
        assert.equal(window.document.activeElement, optimizerButton);
        optimizerButton.click();
        optimizerButton = optimizerHeader().querySelector('button');
        assert.equal(window.document.activeElement, optimizerButton);
        let rows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(rows[0].children[optimizerColumn].textContent, 'zeta');
        assert.equal(rows[1].children[optimizerColumn].textContent, 'alpha');
        assert.equal(optimizerHeader().getAttribute('aria-sort'), 'ascending');
        assert.equal(window.document.querySelectorAll('#trial-thead th[aria-sort]').length, 1);

        optimizerButton.click();
        optimizerButton = optimizerHeader().querySelector('button');
        assert.equal(window.document.activeElement, optimizerButton);
        rows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(rows[0].children[optimizerColumn].textContent, 'alpha');
        assert.equal(rows[1].children[optimizerColumn].textContent, 'zeta');
        assert.equal(optimizerHeader().getAttribute('aria-sort'), 'descending');

        optimizerButton.click();
        optimizerButton = optimizerHeader().querySelector('button');
        assert.equal(window.document.activeElement, optimizerButton);
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

test('sort focus remains on the exact header when column names overlap', () => {
    const { dom, window } = createDom();
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            space: [
                { name: 'overlap', type: 'categorical', choices: ['a', 'b'] },
                { name: 'param · overlap', type: 'integer', min: 0, max: 10 },
                { name: 'metric:overlap', type: 'integer', min: 0, max: 10 },
            ],
            objectives: [{ field: 'loss', obj_type: 'minimize' }],
            trials: [
                {
                    trial_id: 0,
                    params: { overlap: 'a', 'param · overlap': 3, 'metric:overlap': 9 },
                    metrics: { loss: 0.2, overlap: 2 },
                    score_vector: { loss: 0.2 },
                },
                {
                    trial_id: 1,
                    params: { overlap: 'b', 'param · overlap': 4, 'metric:overlap': 8 },
                    metrics: { loss: 0.1, overlap: 1 },
                    score_vector: { loss: 0.1 },
                },
            ],
        });

        const headerButtons = () => [...window.document.querySelectorAll('#trial-thead button')];
        const labelsBeforeSort = headerButtons().map(button => button.textContent);
        const namesBeforeSort = headerButtons().map(button => button.getAttribute('aria-label'));
        assert.equal(new Set(labelsBeforeSort).size, labelsBeforeSort.length);
        assert.equal(new Set(namesBeforeSort).size, namesBeforeSort.length);
        assert.ok(labelsBeforeSort.includes('param · overlap'));
        assert.ok(labelsBeforeSort.includes('param · overlap [2]'));
        assert.ok(labelsBeforeSort.includes('metric · overlap'));

        let metricButton = headerButtons()
            .find(button => button.textContent === 'metric · overlap');
        assert.equal(metricButton.getAttribute('aria-label'), 'Sort trials by metric overlap');
        metricButton.focus();
        metricButton.click();
        metricButton = headerButtons()
            .find(button => button.textContent === 'metric · overlap');
        assert.equal(window.document.activeElement, metricButton);
        assert.equal(metricButton.closest('th').getAttribute('aria-sort'), 'ascending');
        assert.equal(window.document.querySelectorAll('#trial-thead th[aria-sort]').length, 1);

        const labels = headerButtons().map(button => button.textContent);
        const parameterColumn = labels.indexOf('param · overlap');
        const metricColumn = labels.indexOf('metric · overlap');
        const rows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(rows[0].children[0].textContent, '1');
        assert.equal(rows[0].children[parameterColumn].textContent, 'b');
        assert.equal(rows[0].children[metricColumn].textContent, '1');
        assert.equal(rows[1].children[0].textContent, '0');
        assert.equal(rows[1].children[parameterColumn].textContent, 'a');
        assert.equal(rows[1].children[metricColumn].textContent, '2');

        window.sortTable('metric:overlap');
        assert.equal(window.S.sortColumn.key, 'metric:overlap');
        assert.equal(window.S.sortAsc, false);
        assert.equal(window.document.querySelectorAll('#trial-thead th[aria-sort]').length, 1);
        const descendingRows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(descendingRows[0].children[0].textContent, '0');
    } finally {
        dom.window.close();
    }
});

test('replacement checkpoints clear sort state for removed columns', () => {
    const { dom, window } = createDom();
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            space: [{ name: 'old', type: 'integer', min: 0, max: 1 }],
            objectives: [{ field: 'loss', obj_type: 'minimize' }],
            trials: [{
                trial_id: 0,
                params: { old: 0 },
                metrics: { loss: 0 },
                score_vector: { loss: 0 },
            }],
        });
        window.sortTable('param:old');
        assert.equal(window.S.sortColumn.key, 'param:old');

        const trials = Array.from({ length: 1001 }, (_, index) => ({
            trial_id: 10_000 + index,
            params: { current: index },
            metrics: { loss: index },
            score_vector: { loss: index },
        }));
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            space: [{ name: 'current', type: 'integer', min: 0, max: 1000 }],
            objectives: [{ field: 'loss', obj_type: 'minimize' }],
            trials,
        });

        assert.equal(window.S.sortCol, null);
        assert.equal(window.S.sortColumn, null);
        assert.equal(window.S.sortAsc, true);
        assert.equal(window.document.querySelectorAll('#trial-thead th[aria-sort]').length, 0);
        const rows = [...window.document.querySelectorAll('#trial-tbody tr')];
        assert.equal(rows.length, 1000);
        assert.equal(rows[0].children[0].textContent, '10001');
        assert.equal(rows.at(-1).children[0].textContent, '11000');
    } finally {
        dom.window.close();
    }
});

test('long Pareto axis names can wrap and shrink on narrow viewports', () => {
    const { dom, window } = createDom();
    try {
        const style = window.document.createElement('style');
        style.textContent = readFileSync(STYLES_CSS, 'utf8');
        window.document.head.appendChild(style);
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            space: [],
            objectives: [
                { field: 'validation_cross_entropy_loss', obj_type: 'minimize' },
                { field: 'inference_latency_milliseconds', obj_type: 'minimize' },
            ],
            trials: [{
                trial_id: 0,
                metrics: {
                    validation_cross_entropy_loss: 0.25,
                    inference_latency_milliseconds: 12,
                },
                score_vector: {
                    validation_cross_entropy_loss: 0.25,
                    inference_latency_milliseconds: 12,
                },
                rank: 0,
                pareto_front: 0,
            }],
        });

        const selectors = window.document.querySelector('.axis-selectors');
        assert.equal(window.getComputedStyle(selectors).flexWrap, 'wrap');
        for (const label of selectors.querySelectorAll('label')) {
            const labelStyle = window.getComputedStyle(label);
            assert.equal(labelStyle.minWidth, '0');
            assert.equal(labelStyle.maxWidth, '100%');
        }
        for (const select of selectors.querySelectorAll('select')) {
            const selectStyle = window.getComputedStyle(select);
            assert.equal(selectStyle.minWidth, '0');
            assert.equal(selectStyle.maxWidth, '100%');
        }
    } finally {
        dom.window.close();
    }
});

test('objective actions can wrap and shrink under zoom', () => {
    const { dom, window } = createDom();
    try {
        const style = window.document.createElement('style');
        style.textContent = readFileSync(STYLES_CSS, 'utf8');
        window.document.head.appendChild(style);
        const actions = window.document.querySelector('#objectives-card .card-header > div');
        const actionsStyle = window.getComputedStyle(actions);
        assert.equal(actionsStyle.flexWrap, 'wrap');
        assert.equal(actionsStyle.minWidth, '0');
        assert.equal(actionsStyle.maxWidth, '100%');
        for (const input of window.document.querySelectorAll(
            '#connect-panel input[type="text"], #connect-panel input[type="password"]',
        )) {
            const inputStyle = window.getComputedStyle(input);
            assert.equal(inputStyle.minWidth, '0');
        }
    } finally {
        dom.window.close();
    }
});

test('secondary and canvas text retain WCAG AA contrast with safety margin', () => {
    const css = readFileSync(STYLES_CSS, 'utf8');
    const app = readFileSync(APP_JS, 'utf8');
    const token = name => {
        const match = css.match(new RegExp(`${name}:\\s*(#[0-9a-f]{6})`, 'i'));
        assert.ok(match, `missing ${name} color token`);
        return match[1];
    };
    const luminance = hex => {
        const channels = hex.match(/[0-9a-f]{2}/gi).map(channel => {
            const value = Number.parseInt(channel, 16) / 255;
            return value <= 0.04045
                ? value / 12.92
                : ((value + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
    };
    const contrast = (foreground, background) => {
        const values = [luminance(foreground), luminance(background)]
            .sort((left, right) => right - left);
        return (values[0] + 0.05) / (values[1] + 0.05);
    };

    const backgrounds = ['--bg-0', '--bg-1', '--bg-2', '--bg-3'];
    const assertReadable = (foreground, context) => {
        for (const background of backgrounds) {
            const ratio = contrast(foreground, token(background));
            assert.ok(
                ratio >= 4.75,
                `${context} ${foreground} on ${background} is only ${ratio}:1`,
            );
        }
    };

    assertReadable(token('--text-2'), 'secondary text');

    // Canvas text does not inherit CSS variables. Exercise every chart renderer
    // and capture the effective fillStyle at each real fillText call so dynamic
    // or refactored color assignments cannot silently evade this regression.
    const { dom, window, canvasCalls } = createDom();
    try {
        for (const canvas of window.document.querySelectorAll('canvas')) {
            Object.defineProperty(canvas.parentElement, 'clientWidth', {
                configurable: true,
                value: 640,
            });
        }
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            space: [
                { name: 'learning_rate', type: 'real', min: 0.001, max: 0.1 },
                { name: 'optimizer', type: 'categorical', choices: ['adam', 'sgd'] },
            ],
            objectives: [
                { field: 'loss', obj_type: 'minimize' },
                { field: 'latency', obj_type: 'minimize' },
            ],
            trials: [{
                trial_id: 0,
                params: { learning_rate: 0.01, optimizer: 'adam' },
                metrics: { loss: 0.25, latency: 12 },
                score_vector: { loss: 0.25, latency: 12 },
                rank: 0,
                pareto_front: 0,
            }],
        });
        window.renderConvergence();
        window.renderPareto();
        window.renderParallel();

        const textCalls = canvasCalls.filter(call => call.op === 'fillText');
        assert.ok(textCalls.length > 0, 'expected rendered canvas text');
        for (const call of textCalls) {
            assert.match(
                call.fillStyle,
                /^#[0-9a-f]{6}$/i,
                `canvas text ${JSON.stringify(call.text)} must use an auditable hex color`,
            );
            assertReadable(call.fillStyle, `canvas text ${JSON.stringify(call.text)}`);
        }
    } finally {
        dom.window.close();
    }
    assert.doesNotMatch(app, /#(?:555(?:555)?|7070a0)\b/i);
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
        assert.equal(trials[1].pareto_front, null);
        assert.equal(trials[1].rank, null);
        assert.equal(trials.at(-1).pareto_front, null);
        assert.equal(trials.at(-1).rank, null);
        window.replaceTrials(trials);
        assert.equal(window.S.rankingPartial, true);
    } finally {
        dom.window.close();
    }
});

test('preview honors inclusive limits, priorities and wide finite TLP spans', () => {
    const { dom, window } = createDom();
    try {
        for (const [type, target, limit] of [['minimize', 0, 1], ['maximize', 1, 0]]) {
            window.S.objectives = [{ field: 'loss', type, target, limit, priority: 2 }];
            for (const [value, expected] of [[target, 0], [limit, 2], [0.5, 1]]) {
                const trial = { metrics: { loss: value } };
                window.rescoreTrialForPreview(trial);
                assert.equal(trial.score_vector.loss, expected);
                assert.equal(window.isTrialFeasible(trial), true);
            }
        }
        window.S.objectives = [{ field: 'loss', type: 'minimize', target: -1e308, limit: 1e308 }];
        const wide = { metrics: { loss: 0 } };
        window.rescoreTrialForPreview(wide);
        assert.equal(wide.score_vector.loss, 0.5);
        window.S.objectives[0].priority = 0;
        window.S.objectives[0].limit = 1;
        const invalid = { metrics: { loss: 2 } };
        window.rescoreTrialForPreview(invalid);
        assert.equal(invalid.score_vector.loss, Infinity);
        assert.equal(window.isTrialFeasible(invalid), false);
        window.S.objectives = [{ field: 'loss', type: 'minimize', group: '__proto__' }];
        window.rescoreTrialForPreview(wide);
        assert.equal(wide.score_vector.__proto__, 0);
    } finally { dom.window.close(); }
});

test('preview preserves the engine scoring of IEEE-754 metric sentinels', () => {
    const { dom, window } = createDom();
    try {
        for (const [type, target, limit, good, bad] of [
            ['minimize', 0, 1, '-inf', 'inf'],
            ['maximize', 1, 0, 'inf', '-inf'],
        ]) {
            window.S.objectives = [{ field: 'loss', type, target, limit, priority: 0 }];
            const trial = { metrics: { loss: good } };
            window.rescoreTrialForPreview(trial);
            assert.equal(trial.score_vector.loss, 0);
            trial.metrics.loss = bad;
            window.rescoreTrialForPreview(trial);
            assert.equal(trial.score_vector.loss, Infinity);
            trial.metrics.loss = 'nan';
            window.rescoreTrialForPreview(trial);
            assert.equal(Number.isNaN(trial.score_vector.loss), true);
        }
        window.S.objectives = [{ field: 'loss', type: 'maximize' }];
        const trial = { metrics: { loss: 'inf' } };
        window.rescoreTrialForPreview(trial);
        assert.equal(trial.score_vector.loss, -Infinity);
        assert.equal(window.isTrialFeasible(trial), false);
    } finally {
        dom.window.close();
    }
});

test('invalid partial TLP configuration leaves preview state unchanged', () => {
    const { dom, window, alerts } = createDom();
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            objectives: [{ field: 'loss', type: 'minimize', priority: 1 }],
            trials: [{ trial_id: 0, metrics: { loss: 3 }, score_vector: { loss: 3 }, rank: 0, pareto_front: 0 }],
        });
        window.S.objectives[0].target = 0;
        window.previewObjectives();
        assert.equal(alerts.length, 1);
        assert.equal(window.S.previewActive, false);
        assert.equal(window.S.trials[0].score_vector.loss, 3);
        assert.equal(window.document.getElementById('preview-badge').style.display, 'none');
    } finally { dom.window.close(); }
});

test('offline reset restores authoritative scores and ranks across repeated previews', async () => {
    const { dom, window } = createDom();
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            objectives: [{ field: 'loss', type: 'minimize', priority: 1 }],
            trials: [{ trial_id: 0, metrics: { loss: 3 }, score_vector: { loss: 3 }, rank: 0, pareto_front: 0 }],
        });
        window.S.objectives[0].priority = 2;
        window.previewObjectives();
        assert.equal(window.S.trials[0].score_vector.loss, 6);
        window.S.objectives[0].priority = 3;
        window.previewObjectives();
        assert.equal(window.S.trials[0].score_vector.loss, 9);
        await window.resetObjectives();
        assert.equal(window.S.objectives[0].priority, 1);
        assert.equal(window.S.trials[0].score_vector.loss, 3);
        assert.equal(window.S.trials[0].rank, 0);
        assert.equal(window.S.previewActive, false);
        assert.equal(window.S.previewOriginalScores, null);
    } finally { dom.window.close(); }
});

test('live preview replay cannot restore ranks from the server objectives', async () => {
    const { dom, window } = createDom();
    const original = [
        { trial_id: 0, metrics: { a: 0, b: 10, c: 0 }, score_vector: { a: 0, b: 10, c: 0 }, rank: 0, pareto_front: 0 },
        { trial_id: 1, metrics: { a: 5, b: 0, c: 0 }, score_vector: { a: 5, b: 0, c: 0 }, rank: 1, pareto_front: 0 },
    ];
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            objectives: ['a', 'b', 'c'].map(field => ({ field, type: 'minimize' })),
            trials: structuredClone(original),
        });
        window.S.mode = 'live';
        window.S.serverUrl = 'http://preview-replay.test';
        window.S.objectives[0].group = 'quality';
        window.S.objectives[1].group = 'quality';
        window.previewObjectives();
        assert.equal(window.S.trials[0].score_vector.quality, 10);
        assert.equal(window.S.trials[0].pareto_front, 1);
        assert.deepEqual(Array.from(window.S.paretoFrontIds), [1]);

        await window.handleEngineEvent({
            type: 'TrialCompleted', trial_id: 0, trial: structuredClone(original[0]),
        });
        assert.equal(window.S.previewActive, true);
        assert.equal(window.S.trials[0].score_vector.quality, 10);
        assert.equal(window.S.trials[0].pareto_front, 1);
        assert.equal(window.S.trials[0].rank, 1);
        assert.equal(window.S.trials[1].pareto_front, 0);
        assert.deepEqual(Array.from(window.S.paretoFrontIds), [1]);
    } finally { dom.window.close(); }
});

test('partial preview keeps existing replay memberships unknown', async () => {
    const { dom, window } = createDom();
    try {
        window.applyCheckpointData({
            format: 'hola-dashboard-export',
            objectives: ['a', 'b', 'c'].map(field => ({ field, type: 'minimize' })),
            trials: Array.from({ length: 2049 }, (_, i) => ({
                trial_id: i, metrics: { a: i, b: 2048-i, c: 0 },
                score_vector: { a: i, b: 2048-i, c: 0 },
            })),
        });
        window.S.mode = 'live';
        window.S.serverUrl = 'http://partial-preview.test';
        window.previewObjectives();
        assert.equal(window.S.rankingPartial, true);
        assert.equal(window.S.trials[17].pareto_front, null);
        await window.handleEngineEvent({
            type: 'TrialCompleted', trial_id: 17,
            trial: {
                trial_id: 17, metrics: { a: 17, b: 2031, c: 0 },
                score_vector: { a: 17, b: 2031, c: 0 },
                rank: 0, pareto_front: 0,
            },
        });
        assert.equal(window.S.rankingPartial, true);
        assert.equal(window.S.trials[17].pareto_front, null);
        assert.equal(window.S.trials[17].rank, null);
        assert.equal(window.S.trials[17].ranking_status, 'partial');
        assert.deepEqual(Array.from(window.S.paretoFrontIds), [0]);
    } finally { dom.window.close(); }
});

test('partial frontier membership stays unknown on import, export replay and live append', () => {
    const { dom, window } = createDom();
    try {
        const trials = Array.from({length: 2049}, (_, i) => ({
            trial_id: i,
            metrics: {a: i, b: 2048-i, c: 0},
            score_vector: {a: i, b: 2048-i, c: 0},
        }));
        window.applyCheckpointData({format: 'hola-dashboard-export', trials});
        const note = window.document.getElementById('ranking-note');
        assert.equal(note.hidden, false);
        assert.match(note.textContent, /memberships and ranks are unknown/);
        assert.equal(window.S.trials.filter(t => t.pareto_front === 0).length, 1);
        assert.equal(window.S.trials.filter(t => t.pareto_front === null).length, 2048);
        const replay = JSON.parse(JSON.stringify(window.S.trials));
        window.applyCheckpointData({format: 'hola-dashboard-export', trials: replay});
        assert.equal(window.S.rankingPartial, true);
        window.upsertTrial({trial_id: 3000, score_vector: {a: 2, b: 2047, c: 0}, rank: 0, pareto_front: 0});
        assert.equal(window.S.trials.at(-1).pareto_front, null);
        window.upsertTrial({trial_id: 3001, score_vector: {a: -1, b: 3000, c: 0}, rank: 0, pareto_front: 0});
        assert.equal(window.S.trials.at(-1).pareto_front, 0);
        assert.equal(window.S.trials.at(-1).rank, null);
        assert.equal(window.S.trials[0].pareto_front, null);
    } finally { dom.window.close(); }
});

test('live reset replays completions racing its authoritative snapshot', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch);
    const server = 'http://reset-race.test';
    const trialsUrl = `${server}/api/trials?sorted_by=index&include_infeasible=true`;
    const original = {
        trial_id: 0, metrics: { loss: 10 }, score_vector: { loss: 10 },
        rank: 0, pareto_front: 0,
    };
    const completion = {
        type: 'TrialCompleted', trial_id: 1,
        trial: {
            trial_id: 1, metrics: { loss: 1 }, score_vector: { loss: 1 },
            rank: 0, pareto_front: 0,
        },
    };
    try {
        window.S.mode = 'live';
        window.S.serverUrl = server;
        window.S.objectives = [{ field: 'loss', type: 'minimize', priority: 1 }];
        window.S.serverObjectives = JSON.parse(JSON.stringify(window.S.objectives));
        window.replaceTrials([structuredClone(original)]);
        window.S.objectives[0].priority = 2;
        window.previewObjectives();
        const resetting = window.resetObjectives();
        (await router.waitFor(`${server}/api/event_cursor`))
            .resolve(jsonResponse({ last_event_id: '5' }));
        (await router.waitFor(`${server}/api/space`))
            .resolve(jsonResponse({ params: [] }));
        (await router.waitFor(`${server}/api/objectives`))
            .resolve(jsonResponse({ objectives: [{ field: 'loss', type: 'minimize', priority: 1 }] }));
        const snapshot = await router.waitFor(trialsUrl);

        // A previously dispatched completion can finish during the GET. A
        // snapshot captured before it must restart replay from the watermark,
        // rather than permanently erase that completion after its cursor moves.
        await window.handleEngineEvent(structuredClone(completion));
        window.S.lastEventId = '6';
        assert.deepEqual(Array.from(window.S.trials, trial => trial.trial_id), [0, 1]);
        snapshot.resolve(jsonResponse([structuredClone(original)]));
        await resetting;
        const stream = await router.waitFor(`${server}/api/events`);
        assert.equal(stream.options.headers.get('Last-Event-ID'), '5');
        const encoded = new TextEncoder().encode(
            `id: 6\ndata: ${JSON.stringify(completion)}\n\n`,
        );
        let delivered = false;
        stream.resolve({
            ok: true, status: 200,
            body: {
                getReader() {
                    return {
                        read() {
                            if (delivered) return new Promise(() => {});
                            delivered = true;
                            return Promise.resolve({ value: encoded, done: false });
                        },
                    };
                },
            },
        });
        for (let i = 0; i < 20 && window.S.trials.length < 2; i++) {
            await new Promise(resolvePromise => setImmediate(resolvePromise));
        }
        assert.deepEqual(Array.from(window.S.trials, trial => trial.trial_id), [0, 1]);
        assert.equal(window.S.trials[1].score_vector.loss, 1);
        assert.equal(window.S.previewActive, false);
        assert.equal(window.S.objectives[0].priority, 1);
        assert.equal(window.S.lastEventId, '6');
    } finally {
        window.stopStream();
        dom.window.close();
    }
});

test('pending reset and apply cannot overwrite a newly loaded checkpoint', async () => {
    for (const action of ['resetObjectives', 'applyObjectives']) {
        const pending = deferred();
        const { dom, window } = createDom(() => pending.promise);
        try {
            window.S.mode = 'live';
            window.S.serverUrl = 'http://old-server.test';
            window.S.objectives = [{ field: 'loss', type: 'minimize', priority: 1 }];
            const running = window[action]();
            window.S.connectionGeneration++;
            window.applyCheckpointData({
                format: 'hola-dashboard-export',
                objectives: [{ field: 'offline', type: 'minimize', priority: 1 }],
                trials: [{ trial_id: 99, metrics: { offline: 7 }, score_vector: { offline: 7 }, rank: 0, pareto_front: 0 }],
            });
            pending.resolve(jsonResponse(action === 'resetObjectives'
                ? [{trial_id: 0, score_vector: {loss: -1}, rank: 0, pareto_front: 0}]
                : {status: 'ok'}));
            await running;
            assert.equal(window.S.mode, 'offline');
            assert.equal(window.S.trials[0].trial_id, 99);
            assert.equal(window.S.objectives[0].field, 'offline');
        } finally { dom.window.close(); }
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

// ---------------------------------------------------------------------------
// Credentials, redirects, and bounds on what a server can make the page hold.
// ---------------------------------------------------------------------------

const tick = () => new Promise(resolvePromise => setImmediate(resolvePromise));

// Start a connection and return its first request. The connection itself is
// left waiting on that request; closing the window discards it.
async function firstRequest(window, router, server, token = '', occurrence = 0) {
    window.document.getElementById('server-url').value = server;
    window.document.getElementById('api-token').value = token;
    void window.connectToServer();
    return router.waitFor(`${server}/api/event_cursor`, occurrence);
}

function authorization(request) {
    return request.options.headers.get('Authorization');
}

function streamBody(...chunks) {
    const pending = chunks.map(chunk => new TextEncoder().encode(chunk));
    const state = { cancelled: false, reads: 0 };
    return {
        state,
        body: {
            getReader() {
                return {
                    read() {
                        state.reads++;
                        if (pending.length === 0) return Promise.resolve({ done: true });
                        return Promise.resolve({ value: pending.shift(), done: false });
                    },
                    cancel() {
                        state.cancelled = true;
                        return Promise.resolve();
                    },
                };
            },
        },
    };
}

test('a token in the query string is refused, stripped and reported', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(
        router.fetch, 'http://hola.test:8000/index.html?token=query-secret&view=1',
    );
    try {
        assert.equal(window.location.href, 'http://hola.test:8000/index.html?view=1');
        const notice = window.document.getElementById('connection-notice');
        assert.equal(notice.hidden, false);
        assert.match(notice.textContent, /\?token=/);
        assert.match(notice.textContent, /rotate/);
        assert.equal(notice.textContent.includes('query-secret'), false);
        assert.equal(window.document.getElementById('token-status').textContent, '');

        const request = await firstRequest(window, router, 'http://hola.test:8000');
        assert.equal(authorization(request), null);
    } finally {
        dom.window.close();
    }
});

test('a fragment token is erased at load and bound to the server that served the page', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(
        router.fetch, 'http://hola.test:8000/#token=frag%2Bsecret%20x&panel=pareto',
    );
    try {
        assert.equal(window.location.href, 'http://hola.test:8000/#panel=pareto');
        assert.equal(router.requests.size, 0, 'no request may precede erasing the token');
        const status = window.document.getElementById('token-status');
        assert.equal(status.textContent, 'Token set for http://hola.test:8000');
        assert.equal(window.document.getElementById('api-token').value, '');
        assert.equal(window.document.documentElement.outerHTML.includes('frag+secret'), false);

        // An empty URL field falls back to localhost, which is not the server
        // the link named: the token must not follow it there.
        window.document.getElementById('server-url').value = '';
        void window.connectToServer();
        const local = await router.waitFor('http://localhost:8000/api/event_cursor');
        assert.equal(authorization(local), null);
        assert.equal(status.textContent, '');

        // Dropped means dropped: returning to the original origin does not
        // resurrect it.
        const back = await firstRequest(window, router, 'http://hola.test:8000');
        assert.equal(authorization(back), null);
    } finally {
        dom.window.close();
    }
});

test('a fragment token follows ?server= and the page auto-connects with it', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(
        router.fetch, 'http://dash.test/?server=http%3A%2F%2Fapi.test%3A8000#token=link-token',
    );
    try {
        assert.equal(window.location.href, 'http://dash.test/?server=http%3A%2F%2Fapi.test%3A8000');
        const request = await router.waitFor('http://api.test:8000/api/event_cursor');
        assert.equal(authorization(request), 'Bearer link-token');
        // The dashboard's own origin is a different origin and gets nothing.
        void window.apiFetch('http://dash.test/api/space');
        assert.equal(authorization(await router.waitFor('http://dash.test/api/space')), null);
    } finally {
        dom.window.close();
    }
});

test('a fragment token on a page opened from disk binds to the first connection only', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(
        router.fetch, `${pathToFileURL(INDEX_HTML).href}#token=disk-token`,
    );
    try {
        assert.equal(window.location.hash, '');
        assert.match(
            window.document.getElementById('token-status').textContent, /next server you connect to/,
        );
        const first = await firstRequest(window, router, 'http://first.test');
        assert.equal(authorization(first), 'Bearer disk-token');
        const second = await firstRequest(window, router, 'http://second.test');
        assert.equal(authorization(second), null);
    } finally {
        dom.window.close();
    }
});

test('a token navigated to in the fragment of an open page is captured and erased', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch, 'http://hola.test:8000/');
    try {
        window.location.hash = '#token=late-token';
        for (let i = 0; i < 20 && window.location.hash !== ''; i++) await tick();
        assert.equal(window.location.href, 'http://hola.test:8000/');
        const request = await firstRequest(window, router, 'http://hola.test:8000');
        assert.equal(authorization(request), 'Bearer late-token');
    } finally {
        dom.window.close();
    }
});

test('a typed token leaves the form field and never reaches another origin', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch);
    const field = window.document.getElementById('api-token');
    const status = window.document.getElementById('token-status');
    try {
        const a = await firstRequest(window, router, 'https://a.test', 'token-for-a');
        assert.equal(authorization(a), 'Bearer token-for-a');
        assert.equal(field.value, '', 'the token must not linger in the form');
        assert.equal(status.textContent, 'Token set for https://a.test');

        // While bound to A, no request to another origin carries the token:
        // a different host, a different port, and a scheme downgrade.
        for (const url of ['https://b.test/api/space', 'https://a.test:8443/api/space',
            'http://a.test/api/space']) {
            void window.apiFetch(url);
            assert.equal(authorization(await router.waitFor(url)), null, url);
        }

        // Reconnecting to the same origin keeps it without retyping.
        const again = await firstRequest(window, router, 'https://a.test', '', 1);
        assert.equal(authorization(again), 'Bearer token-for-a');

        // Changing the server without supplying a token drops it.
        const b = await firstRequest(window, router, 'https://b.test');
        assert.equal(authorization(b), null);
        assert.equal(status.textContent, '');

        // A token supplied for the new origin is used there, and only there.
        const c = await firstRequest(window, router, 'https://c.test', 'token-for-c');
        assert.equal(authorization(c), 'Bearer token-for-c');
        void window.apiFetch('https://a.test/api/objectives');
        assert.equal(authorization(await router.waitFor('https://a.test/api/objectives')), null);

        window.clearApiToken();
        void window.apiFetch('https://c.test/api/space');
        assert.equal(authorization(await router.waitFor('https://c.test/api/space')), null);
        assert.equal(status.textContent, '');
    } finally {
        dom.window.close();
    }
});

test('a token sent over unencrypted HTTP beyond this machine raises a visible warning', async () => {
    const router = requestRouter();
    const { dom, window, alerts } = createDom(router.fetch);
    const notice = window.document.getElementById('connection-notice');
    try {
        for (const server of ['http://127.0.0.1:8000', 'http://localhost:8000',
            'http://[::1]:8000', 'https://remote.test']) {
            assert.equal(authorization(await firstRequest(window, router, server, 't')), 'Bearer t');
            assert.equal(notice.hidden, true, server);
        }
        // Without a token there is nothing to expose.
        await firstRequest(window, router, 'http://remote.test:8000');
        assert.equal(notice.hidden, true);
        // The warning informs; it does not block the request.
        const remote = await firstRequest(window, router, 'http://remote.test:8000', 't', 1);
        assert.equal(authorization(remote), 'Bearer t');
        assert.equal(notice.hidden, false);
        assert.match(notice.textContent, /not encrypted/);
        assert.match(notice.textContent, /http:\/\/remote\.test:8000/);
        window.clearApiToken();
        assert.equal(notice.hidden, true);

        // A URL without an http(s) origin cannot hold a token and is refused.
        window.document.getElementById('server-url').value = 'remote.test:8000';
        await window.connectToServer();
        assert.match(alerts.at(-1), /must start with http/);
    } finally {
        dom.window.close();
    }
});

test('every API request, including the event stream, refuses redirects', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch);
    const server = 'http://redirect.test';
    try {
        const cursor = await firstRequest(window, router, server);
        assert.equal(cursor.options.redirect, 'error');

        window.S.mode = 'live';
        window.S.objectives = [{ field: 'loss', type: 'minimize', priority: 1 }];
        void window.applyObjectives();
        const patch = await router.waitFor(`${server}/api/objectives`);
        assert.equal(patch.options.method, 'PATCH');
        assert.equal(patch.options.redirect, 'error');

        window.startStream();
        const stream = await router.waitFor(`${server}/api/events`);
        assert.equal(stream.options.redirect, 'error');
    } finally {
        window.stopStream();
        dom.window.close();
    }
});

test('REST bodies are counted while they arrive and cut off at the cap', async () => {
    let response;
    const { dom, window } = createDom(() => Promise.resolve(response));
    const url = 'http://cap.test/api/space';
    try {
        // No Content-Length at all: the cap still applies to the bytes read.
        const flood = streamBody('{"params":[', '"aaaaaaaaaaaaaaaa",', '"bbbbbbbbbbbbbbbb"]}');
        response = { ok: true, status: 200, body: flood.body };
        await assert.rejects(window.apiJson(url, { maxBytes: 32 }), /exceeds the/);
        assert.equal(flood.state.cancelled, true);
        assert.equal(flood.state.reads, 3, 'reading stops at the chunk that crosses the cap');

        // An honest oversized Content-Length is rejected before reading.
        const declared = streamBody('{}');
        response = {
            ok: true, status: 200, body: declared.body,
            headers: new Headers({ 'Content-Length': '1000' }),
        };
        await assert.rejects(window.apiJson(url, { maxBytes: 32 }), /exceeds the/);
        assert.equal(declared.state.reads, 0);
        assert.equal(declared.state.cancelled, true);

        // Within the cap, a body split inside a multi-byte character parses.
        const bytes = new TextEncoder().encode('{"name":"é"}');
        const pieces = [bytes.slice(0, 10), bytes.slice(10)];
        response = {
            ok: true, status: 200,
            body: { getReader: () => ({ read: async () => (pieces.length
                ? { value: pieces.shift(), done: false } : { done: true }) }) },
        };
        const parsed = await window.apiJson(url, { maxBytes: 32 });
        assert.equal(parsed.ok, true);
        assert.equal(parsed.data.name, 'é');

        // An oversized error body is discarded; the status still comes back.
        response = { ok: false, status: 500, body: streamBody('x'.repeat(64)).body };
        assert.deepEqual({ ...await window.apiJson(url, { maxBytes: 32 }) },
            { ok: false, status: 500, data: null });
    } finally {
        dom.window.close();
    }
});

test('a REST request that is never answered fails at its deadline', async () => {
    let signal;
    const { dom, window } = createDom((_url, options) => new Promise((_resolve, reject) => {
        signal = options.signal;
        signal.addEventListener('abort', () => reject(new Error('aborted')));
    }));
    try {
        await assert.rejects(
            window.apiJson('http://slow.test/api/space', { timeoutMs: 20 }),
            /did not answer within/,
        );
        assert.equal(signal.aborted, true);
    } finally {
        dom.window.close();
    }
});

test('a body that stalls after the headers also fails at the deadline', async () => {
    const { dom, window } = createDom((_url, options) => Promise.resolve({
        ok: true, status: 200,
        body: { getReader: () => ({
            read: () => new Promise((_resolve, reject) => {
                options.signal.addEventListener('abort', () => reject(new Error('aborted')));
            }),
        }) },
    }));
    try {
        await assert.rejects(
            window.apiJson('http://slow.test/api/trials', { timeoutMs: 20 }),
            /did not answer within/,
        );
    } finally {
        dom.window.close();
    }
});

async function openStream(window, router, server, occurrence = 0) {
    window.S.mode = 'live';
    window.S.serverUrl = server;
    window.startStream();
    return router.waitFor(`${server}/api/events`, occurrence);
}

function endlessBody(...chunks) {
    const pending = chunks.map(chunk => new TextEncoder().encode(chunk));
    return {
        getReader: () => ({
            read: () => (pending.length
                ? Promise.resolve({ value: pending.shift(), done: false })
                : new Promise(() => {})),
        }),
    };
}

test('an event stream record larger than the cap drops the stream and resyncs', async () => {
    for (const oversized of [
        // Never terminated: the buffer would otherwise grow without bound.
        ['data: ' + 'x'.repeat(600_000), 'x'.repeat(600_000)],
        // Terminated, but delivered in one read.
        ['data: ' + 'x'.repeat(1_100_000) + '\n\n'],
    ]) {
        const router = requestRouter();
        const { dom, window } = createDom(router.fetch);
        const server = 'http://flood.test';
        try {
            assert.equal(window.MAX_SSE_RECORD_CHARS, 1024 * 1024);
            const stream = await openStream(window, router, server);
            const active = window.S.sse;
            stream.resolve({ ok: true, status: 200, body: endlessBody(...oversized) });
            for (let i = 0; i < 50 && window.S.sse === active; i++) await tick();
            assert.equal(window.S.sse, null, 'the oversized stream must be stopped');
            assert.equal(active.signal.aborted, true);
            assert.notEqual(window.S.resyncRetryTimer, null, 'a delayed resync must be scheduled');
            assert.equal(router.requests.has(`${server}/api/event_cursor`), false,
                'the resync waits for its delay instead of looping immediately');
            assert.match(window.document.getElementById('sse-dot').className, /disconnected/);
        } finally {
            window.stopStream();
            dom.window.close();
        }
    }
});

test('a burst of more records than one read may apply drops the stream', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch);
    const server = 'http://burst.test';
    const record = id => `id: ${id}\ndata: {"type":"TrialCompleted","trial_id":${id},`
        + `"trial":{"trial_id":${id},"metrics":{"loss":1},"score_vector":{"loss":1}}}\n\n`;
    try {
        const limit = window.MAX_SSE_RECORDS_PER_READ;
        // Exactly the limit is applied in full.
        const within = await openStream(window, router, server);
        const first = window.S.sse;
        within.resolve({
            ok: true, status: 200,
            body: endlessBody(Array.from({ length: limit }, (_, id) => record(id)).join('')),
        });
        for (let i = 0; i < 50 && window.S.trials.length < limit; i++) await tick();
        assert.equal(window.S.trials.length, limit);
        assert.equal(window.S.sse, first, 'a burst within the limit keeps the stream');

        // One more than the limit stops applying records and schedules a resync.
        const beyond = await openStream(window, router, server, 1);
        const second = window.S.sse;
        beyond.resolve({
            ok: true, status: 200,
            body: endlessBody(
                Array.from({ length: limit + 1 }, (_, id) => record(limit + id)).join(''),
            ),
        });
        for (let i = 0; i < 50 && window.S.sse === second; i++) await tick();
        assert.equal(window.S.sse, null);
        assert.equal(window.S.trials.length, 2 * limit, 'the record past the limit is not applied');
        assert.notEqual(window.S.resyncRetryTimer, null);
    } finally {
        window.stopStream();
        dom.window.close();
    }
});

test('events that need a trial fetch run a bounded number of requests', async () => {
    const router = requestRouter();
    const { dom, window } = createDom(router.fetch);
    const server = 'http://fetches.test';
    const trialUrl = id => `${server}/api/trial/${id}?include_infeasible=true`;
    const trialRequests = () => Array.from(router.requests.keys())
        .filter(key => key.startsWith(`${server}/api/trial/`)).length;
    try {
        window.S.mode = 'live';
        window.S.serverUrl = server;
        const limit = window.MAX_EVENT_FETCHES;
        for (let id = 0; id < limit + 20; id++) {
            void window.handleEngineEvent({ type: 'TrialCompleted', trial_id: id });
        }
        await tick();
        assert.equal(trialRequests(), limit, 'no more than the limit may be in flight');
        // The overflow collapses into a single snapshot refresh.
        assert.equal(router.requests.get(`${server}/api/event_cursor`).length, 1);

        // Finishing one request frees one slot.
        (await router.waitFor(trialUrl(0))).resolve(jsonResponse({
            trial_id: 0, metrics: { loss: 1 }, score_vector: { loss: 1 }, rank: 0, pareto_front: 0,
        }));
        for (let i = 0; i < 20 && window.S.trials.length === 0; i++) await tick();
        assert.equal(window.S.trials.length, 1);
        void window.handleEngineEvent({ type: 'TrialCompleted', trial_id: 500 });
        void window.handleEngineEvent({ type: 'TrialCompleted', trial_id: 501 });
        await tick();
        assert.equal(trialRequests(), limit + 1);
        assert.equal(router.requests.has(trialUrl(500)), true);
        assert.equal(router.requests.has(trialUrl(501)), false);

        // A failed fetch frees its slot too and does not surface as an error.
        (await router.waitFor(trialUrl(1))).reject(new Error('network down'));
        await tick();
        void window.handleEngineEvent({ type: 'TrialCompleted', trial_id: 502 });
        await tick();
        assert.equal(router.requests.has(trialUrl(502)), true);
    } finally {
        window.stopStream();
        dom.window.close();
    }
});
