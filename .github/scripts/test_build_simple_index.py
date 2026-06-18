#!/usr/bin/env python3
"""Tests for build_simple_index.

Runnable both as 'python3 .github/scripts/test_build_simple_index.py' and under
pytest. Stdlib-only, no third-party imports.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_simple_index import (  # noqa: E402
    is_package_asset,
    parse_asset_stream,
    render_package_index,
    render_root_index,
)


def test_is_package_asset_includes_wheel_and_sdist():
    wheel = (
        "hola_opt-1.0.1rc6-cp310-abi3-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    assert is_package_asset(wheel) is True
    assert is_package_asset("hola_opt-1.0.1rc6.tar.gz") is True


def test_is_package_asset_excludes_cli_tarballs_and_zips():
    assert is_package_asset("hola-linux-x86_64.tar.gz") is False
    assert is_package_asset("hola-macos-aarch64.tar.gz") is False
    assert is_package_asset("hola-windows-x86_64.zip") is False


def test_render_package_index_escapes_hostile_filename():
    entries = [
        {
            "filename": 'evil"><script>.whl',
            "url": "https://example.com/a?x=1&y=2\"z",
            "sha256": "deadbeef",
        }
    ]
    out = render_package_index(entries)

    # Raw unescaped dangerous substrings must NOT appear.
    assert "<script>" not in out
    # The raw double-quote from the filename/url must not leak into the markup
    # unescaped: every literal " in the source values should be escaped.
    assert 'x=1&y=2"z' not in out
    assert 'evil"><script>' not in out

    # Escape sequences must be present.
    assert "&amp;" in out
    assert "&lt;" in out
    assert "&quot;" in out

    # The sha256 fragment must be present in the href.
    assert "#sha256=deadbeef" in out


def test_render_root_index_contains_anchor():
    out = render_root_index()
    assert '<a href="hola-opt/">hola-opt</a>' in out


def test_parse_asset_stream_flattens_multiline_pretty_printed_tags():
    # Mimics jq's pretty-printed output for TWO tags: two indented multi-line
    # arrays concatenated with a newline between them. The lone '[' line means
    # the stream cannot be parsed line by line; raw_decode handles it.
    stream = (
        "[\n"
        '  {\n'
        '    "name": "hola_opt-9.9-cp310-abi3-linux_x86_64.whl",\n'
        '    "url": "https://example.com/wheel.whl"\n'
        "  },\n"
        '  {\n'
        '    "name": "hola-linux-x86_64.tar.gz",\n'
        '    "url": "https://example.com/cli.tar.gz"\n'
        "  }\n"
        "]\n"
        "[\n"
        '  {\n'
        '    "name": "hola_opt-9.9.tar.gz",\n'
        '    "url": "https://example.com/sdist.tar.gz"\n'
        "  }\n"
        "]\n"
    )
    assets = parse_asset_stream(stream)
    assert [a["name"] for a in assets] == [
        "hola_opt-9.9-cp310-abi3-linux_x86_64.whl",
        "hola-linux-x86_64.tar.gz",
        "hola_opt-9.9.tar.gz",
    ]


def test_parse_asset_stream_empty_arrays_yield_empty_list():
    assert parse_asset_stream("[]\n[]") == []


def _run_all():
    tests = [
        test_is_package_asset_includes_wheel_and_sdist,
        test_is_package_asset_excludes_cli_tarballs_and_zips,
        test_render_package_index_escapes_hostile_filename,
        test_render_root_index_contains_anchor,
        test_parse_asset_stream_flattens_multiline_pretty_printed_tags,
        test_parse_asset_stream_empty_arrays_yield_empty_list,
    ]
    for test in tests:
        test()
        print(f"  {test.__name__} OK")
    print("OK")


if __name__ == "__main__":
    _run_all()
