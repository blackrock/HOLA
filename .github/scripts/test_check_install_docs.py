#!/usr/bin/env python3
"""Tests for check_install_docs, build_install_requirements, check_pypi_namespace.

Runnable both as 'python3 .github/scripts/test_check_install_docs.py' and under
pytest. Stdlib-only, no third-party imports.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_install_docs as check  # noqa: E402
import check_pypi_namespace as watch  # noqa: E402
from build_install_requirements import (  # noqa: E402
    render_requirements,
    wheel_digests,
)

INDEX = "https://blackrock.github.io/HOLA/simple/"
DIGEST = "ab" * 32


def _markdown(command: str) -> str:
    return f"Install:\n\n```bash\n{command}\n```\n"


def test_rejects_extra_index_for_private_package():
    for command in (
        f"pip install hola-opt --extra-index-url {INDEX}",
        f"pip install --extra-index-url={INDEX} hola_opt",
        f"pip install --pre hola-opt==1.0.1rc8 --extra-index-url {INDEX}",
        f"pip install hola-opt \\\n    --extra-index-url {INDEX}",
        f"uv pip install hola-opt --extra-index-url {INDEX}",
        f"PIP_EXTRA_INDEX_URL={INDEX} pip install Hola.Opt",
        f"uv pip install --index-strategy unsafe-best-match -i {INDEX} hola-opt",
    ):
        problems = check.check_text("docs/x.md", _markdown(command))
        assert any("additional package index" in p for p in problems), command


def test_rejects_extra_index_in_inline_code_and_config_block():
    inline = f"Run `pip install hola-opt --extra-index-url {INDEX}` to install.\n"
    assert check.check_text("README.md", inline)
    config = f"```ini\n[global]\nextra-index-url = {INDEX}\n```\n"
    assert not check.check_text("README.md", config)
    config_with_install = (
        f"```bash\ncat > pip.conf <<EOF\n[global]\nextra-index-url = {INDEX}\nEOF\n"
        "pip install --require-hashes hola-opt\n```\n"
    )
    assert check.check_text("README.md", config_with_install)


def test_rejects_extra_index_outside_markdown():
    workflow = f"      - run: python -m pip install hola-opt --extra-index-url {INDEX}\n"
    assert check.check_text(".github/workflows/x.yml", workflow)
    requirements = f"--extra-index-url {INDEX}\nnumpy\nhola-opt==1.0.1rc8\n"
    assert check.check_text("requirements.txt", requirements)
    pyproject = (
        f'[tool.uv]\nextra-index-url = ["{INDEX}"]\n\n[project]\ndependencies = ["hola_opt"]\n'
    )
    assert check.check_text("example/pyproject.toml", pyproject)


def test_prose_may_warn_about_the_option():
    prose = (
        "Never add this project's index to an install of `hola-opt` with\n"
        "`--extra-index-url`; pip does not prefer one index over another.\n"
    )
    assert check.check_text("docs/x.md", prose) == []
    comment = "# never install hola-opt with --extra-index-url\n"
    assert check.check_text(".github/workflows/x.yml", comment) == []


def test_extra_index_for_other_packages_is_not_this_checks_business():
    command = "pip install torch --extra-index-url https://download.pytorch.org/whl/cpu"
    assert check.check_text("docs/x.md", _markdown(command)) == []


def test_rejects_named_install_from_public_index():
    for command in (
        "pip install hola-opt",
        "python -m pip install 'hola-opt==1.0.1rc8'",
        "pip install --pre hola-opt",
        "pip install --upgrade 'hola_opt>=1.0'",
        "uv add hola-opt",
        "pipx install hola-opt",
        "pip install hola-opt --find-links https://example.com/wheels/",
        "pip install --index-url https://pypi.org/simple/ hola-opt",
        'pip install "hola-opt @ https://example.com/hola_opt-1.0-py3-none-any.whl"',
    ):
        problems = check.check_text("docs/x.md", _markdown(command))
        assert any("installs hola-opt by name" in p for p in problems), command


def test_accepts_safe_installs():
    for command in (
        check.CANONICAL_INSTALL,
        f"pip install --index-url {INDEX} hola-opt",
        "python -m pip install hola-opt --no-index --find-links dist/",
        'pip download --no-deps --index-url "file://$PWD/index/simple/" hola-opt -d /tmp/x',
        f'pip install "hola-opt @ https://example.com/hola_opt-1.0-py3-none-any.whl#sha256={DIGEST}"',
        "pip install ./hola_opt-1.0.1rc8-cp310-abi3-win_amd64.whl",
        "uv pip install --python .venv/bin/python --force-reinstall dist/*.whl",
        "pip install scikit-learn",
        "uv run --directory hola-py maturin develop",
    ):
        assert check.check_text("docs/x.md", _markdown(command)) == [], command


def test_named_installs_allowed_once_public_index_is_authoritative():
    original = check.PUBLIC_INDEX_IS_AUTHORITATIVE
    check.PUBLIC_INDEX_IS_AUTHORITATIVE = True
    try:
        assert check.check_text("docs/x.md", _markdown("pip install hola-opt")) == []
        mixed = _markdown(f"pip install hola-opt --extra-index-url {INDEX}")
        assert check.check_text("docs/x.md", mixed)
    finally:
        check.PUBLIC_INDEX_IS_AUTHORITATIVE = original


def test_synced_docs_must_carry_the_canonical_command():
    good = {path: _markdown(check.CANONICAL_INSTALL) for path in check.SYNCED_DOCS}
    assert check.check_synced_docs(good) == []
    drifted = dict(good)
    drifted["docs/index.md"] = _markdown(f"pip install --index-url {INDEX} hola-opt")
    assert [p.split(":")[0] for p in check.check_synced_docs(drifted)] == ["docs/index.md"]
    missing = {k: v for k, v in good.items() if k != "README.md"}
    assert [p.split(":")[0] for p in check.check_synced_docs(missing)] == ["README.md"]


def _lock(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_lock_file_must_be_hash_locked_to_the_trusted_index():
    good = _lock(
        "# comment",
        "--require-hashes",
        f"--index-url {INDEX}",
        "--only-binary hola-opt",
        "hola-opt==1.0.1rc8 \\",
        f"    --hash=sha256:{DIGEST} \\",
        f"    --hash=sha256:{'cd' * 32}",
    )
    assert check.check_lock_file("lock.txt", good) == []

    cases = {
        "not pinned": good.replace("hola-opt==1.0.1rc8", "hola-opt>=1.0"),
        "sha256 digest": good.replace(f"--hash=sha256:{DIGEST}", "--hash=sha256:abcd"),
        "--index-url": good.replace(f"--index-url {INDEX}", "--index-url https://pypi.org/simple/"),
        "second package source": good + f"--extra-index-url {INDEX}\n",
        "unexpected requirement": good + f"requests==2.0 --hash=sha256:{DIGEST}\n",
    }
    for expected, text in cases.items():
        problems = check.check_lock_file("lock.txt", text)
        assert any(expected in p for p in problems), (expected, problems)
    unhashed = _lock(f"--index-url {INDEX}", "hola-opt==1.0.1rc8")
    assert any("sha256 digest" in p for p in check.check_lock_file("lock.txt", unhashed))
    assert any("--index-url" in p for p in check.check_lock_file("lock.txt", "hola-opt==1\n"))


def test_generated_lock_file_passes_the_check():
    release = {
        "tag_name": "v9.9.9-rc1",
        "assets": [
            {"name": "hola-linux-x86_64.tar.gz", "digest": f"sha256:{'11' * 32}"},
            {"name": "hola_opt-9.9.9rc1.tar.gz", "digest": f"sha256:{'22' * 32}"},
            {"name": "hola_opt-9.9.9rc1-cp310-abi3-win_amd64.whl", "digest": f"sha256:{'33' * 32}"},
            {
                "name": "hola_opt-9.9.9rc1-cp310-abi3-manylinux_2_28_x86_64.whl",
                "digest": f"sha256:{'44' * 32}",
            },
        ],
    }
    version, wheels = wheel_digests(release)
    assert version == "9.9.9rc1"
    assert [digest for _, digest in wheels] == ["44" * 32, "33" * 32]
    text = render_requirements(release["tag_name"], version, wheels)
    assert check.check_lock_file("lock.txt", text) == []
    assert "22" * 32 not in text and "11" * 32 not in text


def test_generator_refuses_incomplete_releases():
    wheel = "hola_opt-1.0-cp310-abi3-win_amd64.whl"
    for release in (
        {"assets": []},
        {"assets": [{"name": wheel, "digest": None}]},
        {"assets": [{"name": wheel, "digest": "md5:abc"}]},
        {
            "assets": [
                {"name": wheel, "digest": f"sha256:{DIGEST}"},
                {"name": "hola_opt-1.1-cp310-abi3-win_amd64.whl", "digest": f"sha256:{DIGEST}"},
            ]
        },
    ):
        try:
            wheel_digests(release)
        except ValueError:
            continue
        raise AssertionError(f"accepted {release}")


def test_namespace_watch_classification():
    trusted = {"aa" * 32, "bb" * 32}
    assert watch.classify("hola-opt", None, trusted) == []
    ours = [("hola_opt-1.0-cp310-abi3-win_amd64.whl", "aa" * 32)]
    assert watch.classify("hola-opt", ours, trusted) == []
    assert "no files" in watch.classify("hola-opt", [], trusted)[0]
    foreign = ours + [("hola_opt-99.0-py3-none-any.whl", "cc" * 32)]
    alerts = watch.classify("hola-opt", foreign, trusted)
    assert len(alerts) == 1 and "hola_opt-99.0-py3-none-any.whl" in alerts[0]
    assert watch.classify("hola-opt", [("hola_opt-1.0.tar.gz", "")], trusted)


def test_namespace_watch_checks_lock_file_digests():
    trusted = {"aa" * 32}
    assert watch.lock_file_alerts(f"hola-opt==1 --hash=sha256:{'aa' * 32}\n", trusted) == []
    assert watch.lock_file_alerts(f"hola-opt==1 --hash=sha256:{'cc' * 32}\n", trusted)
    assert watch.lock_file_alerts("hola-opt==1\n", trusted)


def test_repository_is_clean():
    root = Path(__file__).resolve().parents[2]
    assert check.check_repository(root) == []


def _run_all():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__} OK")
    print("OK")


if __name__ == "__main__":
    _run_all()
