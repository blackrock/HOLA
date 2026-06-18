#!/usr/bin/env python3
"""Generate a PEP 503 "simple" package index for hola-opt.

Reads a JSON array of release-asset objects ({"name": ..., "url": ...}) from
stdin, keeps only the Python distribution artifacts (wheels and the project
sdist -- NOT the CLI binary tarballs/zips), computes a sha256 for each asset by
downloading it, and writes the per-package and root index HTML files under
index/simple/.

Pure, stdlib-only helpers are exposed for unit testing.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import urllib.request
from hashlib import sha256

# Matches the project source distribution (e.g. 'hola_opt-1.0.1rc6.tar.gz' or
# 'hola-opt-1.0.tar.gz') but NOT the CLI binary tarballs like
# 'hola-linux-x86_64.tar.gz'.
PROJECT_SDIST_RE = re.compile(r"^hola[_-]opt-.*\.tar\.gz$")


def parse_asset_stream(text: str) -> list:
    """Parse a stream of one-or-more concatenated JSON values into one list.

    ``text`` is the concatenation of JSON values as emitted per release tag by
    ``gh ... --jq '[.assets[] | {name, url}]'``. Each value is a JSON array of
    asset objects, but jq may pretty-print each array across multiple lines, and
    the values are separated by arbitrary whitespace/newlines. This decodes
    each value with ``json.JSONDecoder().raw_decode`` and flattens the
    arrays into a single list of asset dicts.

    - List values are flattened (their elements extend the result).
    - Dict values are appended directly.
    - Empty arrays contribute nothing.
    """
    decoder = json.JSONDecoder()
    result: list = []
    idx = 0
    length = len(text)
    while idx < length:
        # Skip whitespace between concatenated JSON values.
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        value, end = decoder.raw_decode(text, idx)
        if isinstance(value, list):
            result.extend(value)
        elif isinstance(value, dict):
            result.append(value)
        idx = end
    return result


def is_package_asset(name: str) -> bool:
    """Return True for Python distribution artifacts (wheels and project sdist).

    Excludes CLI binary tarballs and .zip files.
    """
    return name.endswith(".whl") or PROJECT_SDIST_RE.fullmatch(name) is not None


def render_package_index(entries) -> str:
    """Render the PEP 503 per-package index page.

    ``entries`` is a list of dicts each with keys 'filename', 'url', 'sha256'.
    """
    lines = ["<!DOCTYPE html><html><body>"]
    for entry in entries:
        url = html.escape(entry["url"], quote=True)
        filename = html.escape(entry["filename"])
        digest = entry["sha256"]
        lines.append(f'  <a href="{url}#sha256={digest}">{filename}</a><br/>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def render_root_index() -> str:
    """Render the PEP 503 root index page."""
    return (
        "<!DOCTYPE html><html><body>\n"
        '  <a href="hola-opt/">hola-opt</a>\n'
        "</body></html>\n"
    )


def sha256_of_url(url, retries=3) -> str:
    """Download the bytes at ``url`` and return their hex sha256.

    Follows redirects. Retries on failure up to ``retries`` times; raises if all
    attempts fail (never returns a placeholder hash).
    """
    last_exc = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            return sha256(data).hexdigest()
        except Exception as exc:  # noqa: BLE001 - retried then re-raised below
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(2)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exc}")


def main() -> None:
    assets = parse_asset_stream(sys.stdin.read())
    kept = [a for a in assets if is_package_asset(a["name"])]
    if not kept:
        print("No wheel or sdist assets found for the package index", file=sys.stderr)
        sys.exit(1)

    entries = []
    for asset in kept:
        digest = sha256_of_url(asset["url"])
        entries.append(
            {"filename": asset["name"], "url": asset["url"], "sha256": digest}
        )

    pkg_dir = os.path.join("index", "simple", "hola-opt")
    os.makedirs(pkg_dir, exist_ok=True)
    with open(os.path.join(pkg_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(render_package_index(entries))
    with open(
        os.path.join("index", "simple", "index.html"), "w", encoding="utf-8"
    ) as fh:
        fh.write(render_root_index())

    print(f"Generated index with {len(entries)} assets")


if __name__ == "__main__":
    main()
