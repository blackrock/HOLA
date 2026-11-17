#!/usr/bin/env python3
"""Watch the public PyPI namespace of this project's package names.

Anyone can register an unclaimed name on PyPI. This check reports the moment a
watched name is registered, or gains a file, that did not come from this
repository's release workflow, so maintainers can investigate before pointing
users at an install command again.

For every name in ``PRIVATE_PACKAGES`` it asks PyPI for the project's files and
compares each SHA-256 with the digests GitHub recorded for this repository's
release assets:

- not registered: reported, and not an alert;
- registered and every file is one of our release assets: OK;
- registered with no files, or with any file we did not release: ALERT.

It also confirms that every digest in the hash-locked install file is a
release-asset digest.

Exit status: 0 OK, 1 alert, 2 the check could not be completed (network or API
failure). With ``GITHUB_OUTPUT`` set, writes ``alert=true|false``. Stdlib-only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_install_docs import LOCK_FILE, PRIVATE_PACKAGES, normalize  # noqa: E402

PYPI_SIMPLE = "https://pypi.org/simple/{name}/"
PYPI_JSON = "application/vnd.pypi.simple.v1+json"
USER_AGENT = "package-namespace-watch"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CheckFailed(Exception):
    """The check could not be completed; says nothing about the namespace."""


def fetch(url: str, headers: dict[str, str], retries: int = 3) -> tuple[int, bytes]:
    """GET ``url``; return ``(status, body)`` for 200 and 404, else raise."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, b""
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt + 1 < retries:
            time.sleep(5 * (attempt + 1))
    raise CheckFailed(f"GET {url} failed: {last}")


def pypi_files(name: str) -> list[tuple[str, str]] | None:
    """Return ``[(filename, sha256)]`` for a PyPI project, or None if unregistered."""
    status, body = fetch(PYPI_SIMPLE.format(name=normalize(name)), {"Accept": PYPI_JSON})
    if status == 404:
        return None
    try:
        files = json.loads(body)["files"]
        return [(entry["filename"], entry.get("hashes", {}).get("sha256", "")) for entry in files]
    except (ValueError, KeyError, TypeError) as exc:
        raise CheckFailed(f"unexpected PyPI response for {name}: {exc}") from exc


def release_digests(repository: str, token: str | None) -> set[str]:
    """Return the SHA-256 digests of every release asset of ``repository``."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    digests: set[str] = set()
    for page in range(1, 51):
        url = f"https://api.github.com/repos/{repository}/releases?per_page=100&page={page}"
        status, body = fetch(url, headers)
        if status != 200:
            raise CheckFailed(f"cannot list releases of {repository} (HTTP {status})")
        releases = json.loads(body)
        if not releases:
            break
        for release in releases:
            for asset in release.get("assets", []):
                algorithm, _, value = (asset.get("digest") or "").partition(":")
                if algorithm == "sha256" and SHA256_RE.fullmatch(value):
                    digests.add(value)
    return digests


def classify(name: str, files: list[tuple[str, str]] | None, trusted: set[str]) -> list[str]:
    """Return alert messages for one watched name (empty when nothing is wrong)."""
    if files is None:
        return []
    if not files:
        return [f"{name} is registered on PyPI but has no files; we did not register it"]
    return [
        f"{name} on PyPI offers {filename} (sha256 {digest or 'missing'}), "
        "which is not an asset of any release of this repository"
        for filename, digest in files
        if digest not in trusted
    ]


def lock_file_alerts(text: str, trusted: set[str]) -> list[str]:
    """Every digest pinned by the install file must be a release-asset digest."""
    pinned = re.findall(r"--hash=sha256:([0-9a-f]{64})", text)
    if not pinned:
        return [f"{LOCK_FILE} pins no digests"]
    return [
        f"{LOCK_FILE} pins sha256 {digest}, which is not an asset of any release"
        for digest in pinned
        if digest not in trusted
    ]


def main() -> int:
    repository = os.environ.get("GITHUB_REPOSITORY", "blackrock/HOLA")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    root = Path(__file__).resolve().parents[2]
    alerts: list[str] = []
    try:
        trusted = release_digests(repository, token)
        if not trusted:
            raise CheckFailed(f"{repository} has no release assets with sha256 digests")
        for name in PRIVATE_PACKAGES:
            files = pypi_files(name)
            if files is None:
                print(f"{name}: not registered on PyPI (the name is still unclaimed)")
            else:
                print(f"{name}: registered on PyPI with {len(files)} file(s)")
            alerts.extend(classify(name, files, trusted))
        alerts.extend(lock_file_alerts((root / LOCK_FILE).read_text(encoding="utf-8"), trusted))
    except CheckFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"alert={'true' if alerts else 'false'}\n")
    for alert in alerts:
        print(f"ALERT: {alert}", file=sys.stderr)
    if alerts:
        return 1
    print("package namespace OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
