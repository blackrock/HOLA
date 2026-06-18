# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "dashboard"
TESTS_DIR = DASHBOARD / "tests"


def test_dashboard_app_avoids_html_sinks_for_untrusted_data():
    """Fast static guard: app.js must not use any HTML/script-injection sink.

    This is a cheap first line of defence that fails the moment a dangerous sink
    is introduced, independent of the (slower) jsdom DOM test below. The forbidden
    set covers the common DOM XSS sinks and dynamic-code constructs.
    """
    app_js = (DASHBOARD / "app.js").read_text()

    # Plain substring sinks.
    forbidden_substrings = [
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "document.writeln",
        "eval(",
        "new Function",
        "javascript:",
    ]
    for token in forbidden_substrings:
        assert token not in app_js, f"app.js must not use forbidden sink: {token!r}"

    # Inline event-handler injection via setAttribute. The app wires events
    # through addEventListener (and the occasional function-reference handler
    # assignment), never from untrusted data, so setAttribute() of any on*
    # attribute is never legitimate here and is forbidden outright. String-valued
    # on* property assignments are checked separately below.
    assert not re.search(r"""setAttribute\(\s*['"]on\w+['"]""", app_js), (
        "app.js must not setAttribute() an inline event handler"
    )
    # Forbid assigning a *string* to an on* property. A string handler
    # (`el.onerror = "<untrusted>"`) is executed as inline JS, so it is the XSS
    # danger; assigning a function reference or arrow function is safe and is how
    # the app legitimately wires the occasional handler. We therefore flag only
    # on*-assignments whose right-hand side opens a string literal.
    for m in re.finditer(r"""\.(on\w+)\s*=\s*(['"`])""", app_js):
        raise AssertionError(f"app.js assigns a string to {m.group(1)} (inline-handler XSS risk)")


def _ensure_jsdom_installed() -> bool:
    """Return True if dashboard/tests/node_modules/jsdom is present.

    Installs via `npm install` in dashboard/tests if missing. Returns False if
    npm/install is unavailable (e.g. no network) so the caller can skip.
    """
    if (TESTS_DIR / "node_modules" / "jsdom").exists():
        return True
    if shutil.which("npm") is None:
        return False
    try:
        result = subprocess.run(
            ["npm", "install"],
            cwd=TESTS_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return (TESTS_DIR / "node_modules" / "jsdom").exists()


def test_dashboard_renders_untrusted_trial_as_inert_text():
    """Real DOM test: load the dashboard in jsdom, feed it the malicious
    checkpoint, render, and assert the payloads are inert text, not live HTML.

    Drives app.js's actual render path through jsdom so a regression that
    injects untrusted data via innerHTML (instead of textContent) is caught:
    the node script exits non-zero and prints the failed assertions.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available; skipping DOM XSS test")

    if not _ensure_jsdom_installed():
        pytest.skip("jsdom could not be installed (no network?); skipping DOM XSS test")

    script = TESTS_DIR / "xss_dom.test.mjs"
    result = subprocess.run(
        [node, str(script)],
        cwd=TESTS_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"DOM XSS test failed:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_dashboard_dom_test_detects_injection():
    """Revert-check: prove the jsdom DOM test is discriminating, not vacuous.

    A passing XSS regression test is only meaningful if it actually FAILS on a
    vulnerable app.js. This meta-test copies app.js, flips the untrusted
    trial-cell render sink from the safe ``textContent`` to the unsafe
    ``innerHTML``, runs ``xss_dom.test.mjs`` against the patched copy via the
    HOLA_APP_JS override, and asserts the DOM test exits NON-ZERO -- i.e. it
    detects the injected live-HTML sink.

    Skips under the same conditions as the real DOM test (node/jsdom absent) so
    it never spuriously fails where node is unavailable.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available; skipping DOM XSS revert-check")

    if not _ensure_jsdom_installed():
        pytest.skip("jsdom could not be installed (no network?); skipping DOM XSS revert-check")

    app_js_src = (DASHBOARD / "app.js").read_text()

    # The safe sink: untrusted trial cell values are assigned via textContent
    # (inert text). Flip it to innerHTML so the value is parsed as live HTML.
    safe_sink = "td.textContent = fmtCell(getCellValue(t, c));"
    unsafe_sink = "td.innerHTML = fmtCell(getCellValue(t, c));"
    patched, n_subs = re.subn(re.escape(safe_sink), unsafe_sink, app_js_src, count=1)
    assert n_subs > 0, (
        "Expected safe trial-cell render sink not found in app.js; the "
        f"revert-check could not patch the file. Looked for: {safe_sink!r}. "
        "If app.js was refactored, update this revert-check to target the "
        "current untrusted-cell textContent assignment."
    )

    script = TESTS_DIR / "xss_dom.test.mjs"
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - kept open across subprocess use; cleaned up in finally
        mode="w",
        prefix="app_unsafe_",
        suffix=".js",
        dir=str(DASHBOARD),
        delete=False,
    )
    try:
        tmp.write(patched)
        tmp.close()

        env = dict(os.environ)
        env["HOLA_APP_JS"] = tmp.name
        result = subprocess.run(
            [node, str(script)],
            cwd=TESTS_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert result.returncode != 0, (
            "Revert-check FAILED: the DOM XSS test passed against an app.js "
            "patched to use innerHTML for untrusted trial cells. The DOM test "
            "is NOT discriminating -- it would not catch a real regression.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp.name)
