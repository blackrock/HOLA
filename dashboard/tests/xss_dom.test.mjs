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

// Real DOM-rendering XSS regression test for the HOLA dashboard.
//
// Loads dashboard/index.html in jsdom with the page's own app.js executing,
// feeds it the malicious xss-smoke-checkpoint.json trial, drives the app's
// render path, and asserts every untrusted payload is rendered as INERT TEXT
// (via textContent) rather than live HTML. A live-HTML regression (e.g.
// switching a cell to innerHTML) would materialize an <img onerror> or <script>
// node, which this test detects and fails on.
//
// The app.js under test defaults to ../app.js but can be overridden with the
// HOLA_APP_JS env var so a patched copy can be exercised. This is used by the
// automated pytest revert-check (test_dashboard_dom_test_detects_injection in
// hola-py/tests/test_dashboard_security.py), which patches app.js to use an
// unsafe innerHTML sink and asserts this test then exits non-zero -- proving the
// test is discriminating rather than vacuous.

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { JSDOM, VirtualConsole } from 'jsdom';

const HERE = dirname(fileURLToPath(import.meta.url));
const DASHBOARD_DIR = resolve(HERE, '..');

const APP_JS = process.env.HOLA_APP_JS
    ? resolve(process.env.HOLA_APP_JS)
    : resolve(DASHBOARD_DIR, 'app.js');
const INDEX_HTML = resolve(DASHBOARD_DIR, 'index.html');
const CHECKPOINT = resolve(DASHBOARD_DIR, 'xss-smoke-checkpoint.json');

const failures = [];
function check(cond, msg) {
    if (!cond) failures.push(msg);
}

// Build a self-contained HTML document: the real index.html markup with the
// (possibly overridden) dependency-free app.js inlined, so the page renders
// without a server or network access.
function buildHtml() {
    let html = readFileSync(INDEX_HTML, 'utf8');
    // Replace the local <script src="app.js"> with the inlined app source so we
    // control exactly which app.js is executed.
    const appSrc = readFileSync(APP_JS, 'utf8');
    // app.js declares S and its functions with const/function at script top
    // level. function declarations become global, but `const S` does not attach
    // to window, so a bridge script (run in the same global scope) exposes the
    // symbols this test drives. Referencing a name that does not exist would
    // throw, so guard each with typeof.
    const bridge = `
        try { window.S = S; } catch (e) {}
        for (const name of ['renderAll','upsertTrial','computeRanksIfMissing','discoverMetrics','renderTable','setParamNames']) {
            try { if (typeof eval(name) === 'function') window[name] = eval(name); } catch (e) {}
        }
    `;
    html = html.replace(
        /<script\b[^>]*\bsrc=["']app\.js["']><\/script>/i,
        `<script>${appSrc.replace(/<\/script>/gi, '<\\/script>')}</script>\n`
        + `<script>${bridge}</script>`,
    );
    return html;
}

const html = buildHtml();
const checkpoint = JSON.parse(readFileSync(CHECKPOINT, 'utf8'));

const virtualConsole = new VirtualConsole();
// Swallow expected console noise (e.g. failed connect alerts) but surface real
// errors for debugging.
virtualConsole.on('jsdomError', (e) => {
    // Network/connect errors are expected since fetch is stubbed to reject; only
    // print unexpected ones.
    if (!/fetch|network|connect/i.test(String(e.message || e))) {
        console.error('[jsdom error]', e.message || e);
    }
});

const dom = new JSDOM(html, {
    url: pathToFileURL(INDEX_HTML).href,
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    virtualConsole,
    beforeParse(window) {
        // Stub fetch so app.js load-time/startup code never hits the network or
        // throws an unhandled rejection.
        window.fetch = () =>
            Promise.resolve({
                ok: false,
                status: 0,
                json: () => Promise.resolve({}),
                text: () => Promise.resolve(''),
                body: null,
            });
        // requestAnimationFrame may be used by renderAll for deferred charts.
        window.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 0);
        window.cancelAnimationFrame = (id) => clearTimeout(id);
        // Silence alert/confirm which the app may call on connect failure.
        window.alert = () => {};
        window.confirm = () => true;
    },
});

const { window } = dom;
const { document } = window;

function run() {
    const S = window.S;
    check(S && typeof S === 'object', 'app.js did not expose global state object S');
    if (!S) return;

    // Basic accessibility contract for controls and non-text visualizations.
    check(document.getElementById('status-bar')?.getAttribute('aria-live') === 'polite',
        'status updates must be exposed through an aria-live region');
    check(!!document.querySelector('#server-url[aria-label]'),
        'server URL control needs an accessible name');
    check(!!document.querySelector('#api-token[aria-label]'),
        'API token control needs an accessible name');
    check(!!document.querySelector('#trial-table caption'),
        'trial table needs an accessible caption');
    check(document.querySelectorAll('canvas[role="img"][aria-label]').length >= 2,
        'canvas charts need text alternatives');
    check(document.querySelectorAll('script[src^="http"], link[href^="http"]').length === 0,
        'dashboard must not depend on remote scripts, styles, or fonts');
    const csp = document.querySelector('meta[http-equiv="Content-Security-Policy"]')
        ?.getAttribute('content') || '';
    check(csp.includes("default-src 'self'"), 'dashboard must declare a default-src CSP');
    check(csp.includes("script-src 'self'"), 'dashboard CSP must restrict scripts to self');
    check(!/script-src[^;]*(?:'unsafe-inline'|'unsafe-eval')/.test(csp),
        'dashboard CSP must not allow inline/eval scripts');

    const lb = checkpoint.leaderboard;
    check(lb && Array.isArray(lb.trials) && lb.trials.length > 0,
        'checkpoint has no leaderboard trials');
    const raw = lb.trials[0];

    // Map the persisted trial to the dashboard's CompletedTrial shape, mirroring
    // loadCheckpointFile, then drive the public render path.
    const trial = {
        trial_id: raw.trial_id ?? 0,
        params: raw.candidate ?? raw.params ?? {},
        metrics: raw.raw_metrics ?? raw.metrics ?? {},
        scores: raw.scores ?? {},
        score_vector: raw.score_vector ?? {},
        rank: raw.rank,
        pareto_front: raw.pareto_front,
        completed_at: raw.timestamp ?? 0,
    };

    S.trials = [];
    window.upsertTrial(trial);
    if (typeof window.computeRanksIfMissing === 'function') {
        window.computeRanksIfMissing(S.trials);
    }
    // Derive param/metric columns the way the offline loader does.
    if (typeof window.setParamNames === 'function') window.setParamNames(Object.keys(trial.params));
    else S.paramNames = Object.keys(trial.params);
    S.space = S.paramNames.map((name) => ({
        name, type: 'real', min: 0, max: 1, scale: 'linear',
    }));
    S.objectives = Array.isArray(checkpoint.config?.objectives)
        ? checkpoint.config.objectives : [];
    if (typeof window.discoverMetrics === 'function') window.discoverMetrics();
    S.mode = 'offline';

    check(typeof window.renderAll === 'function', 'app.js did not expose renderAll()');
    window.renderAll();

    const sortableHeaders = [...document.querySelectorAll('#trial-thead th')];
    check(sortableHeaders.length > 0
        && sortableHeaders.every((th) => th.querySelector('button')?.tabIndex === 0),
        'sortable table headers must contain keyboard-focusable native buttons');
    check(sortableHeaders.every((th) => th.getAttribute('role') === null),
        'sortable table cells must retain their native columnheader role');
    check(document.querySelectorAll('#trial-thead th[aria-sort]').length <= 1,
        'aria-sort must appear on at most the actively sorted column header');

    // Collect the untrusted payload strings from the checkpoint.
    const payloads = [];
    for (const obj of [trial.params, trial.metrics]) {
        for (const [k, v] of Object.entries(obj)) {
            payloads.push(String(k));
            payloads.push(String(v));
        }
    }
    for (const objective of S.objectives) payloads.push(String(objective.field));
    const htmlPayloads = payloads.filter((p) => /[<>]/.test(p));
    check(htmlPayloads.length > 0, 'fixture contained no HTML-like payloads to test');

    const tbody = document.getElementById('trial-tbody');
    const thead = document.getElementById('trial-thead');
    const objectives = document.getElementById('objectives-list');
    check(!!tbody, 'trial-tbody not found in DOM');

    // 1) No live nodes may have been created from the payloads.
    check(document.querySelector('img[onerror]') === null,
        'XSS: an <img onerror> element was created from untrusted data');
    check(document.querySelector('svg') === null,
        'XSS: an <svg> element was created from untrusted data');

    // No <script> element should originate from the payload. Count scripts whose
    // text contains the alert payloads (the app legitimately has its own inlined
    // script blocks; those do not contain alert(4)).
    const injectedScripts = [...document.querySelectorAll('script')]
        .filter((s) => /alert\(\d\)/.test(s.textContent || ''));
    check(injectedScripts.length === 0,
        'XSS: a <script> element containing the payload was created');

    // 2) The payloads must appear somewhere as literal text (rendered inert).
    // The table renders param/metric *values* as cells; verify at least the
    // metric value payload "<script>alert(4)</script>" shows up as text and not
    // as a live node.
    const renderedText = (tbody.textContent || '')
        + (thead.textContent || '')
        + (objectives?.textContent || '');
    for (const p of htmlPayloads) {
        check(renderedText.includes(p), `untrusted payload was not rendered as inert text: ${p}`);
    }

    // 3) Defensive: the table region must contain zero element nodes that came
    // from parsing a payload as HTML. Re-check by scanning innerHTML of the body
    // for an unescaped payload tag; if a value were injected as HTML the literal
    // "<img" / "<script" would appear unescaped inside an element rather than as
    // escaped text "&lt;img".
    const tbodyHtml = tbody.innerHTML || '';
    check(!/<img\b/i.test(tbodyHtml),
        'XSS: tbody innerHTML contains an unescaped <img tag');
    check(!/<script\b/i.test(tbodyHtml),
        'XSS: tbody innerHTML contains an unescaped <script tag');
    check(!/<svg\b/i.test(tbodyHtml),
        'XSS: tbody innerHTML contains an unescaped <svg tag');
}

try {
    run();
} catch (e) {
    failures.push('test threw: ' + (e && e.stack ? e.stack : e));
}

if (failures.length > 0) {
    console.error('XSS DOM test FAILED:');
    for (const f of failures) console.error('  - ' + f);
    process.exit(1);
}
console.log('XSS DOM test PASSED: untrusted payloads rendered as inert text.');
process.exit(0);
