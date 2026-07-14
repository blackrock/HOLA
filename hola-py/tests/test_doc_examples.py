# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Test that Python code blocks in documentation compile and (where possible) execute."""

import re
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent.parent
DOCS_DIR = ROOT_DIR / "docs"
PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
LOCAL_LINK_RE = re.compile(r"(?<!!)\[[^]]+\]\(([^)#]+)(?:#[^)]*)?\)")

# Patterns that indicate a block references undefined names, needs a server,
# or is pseudo-code (not real Python). These blocks are only syntax-checked.
SYNTAX_ONLY_MARKERS = [
    # Server/remote references
    "Study.connect(",
    "study.serve(",
    "study.stop(",
    "remote.",
    # Undefined function calls
    "train_model(",
    "train(",
    "measure_latency(",
    "my_function(",
    # Environment variable / stdlib patterns in worker scripts
    "os.environ[",
    "urllib.request",
    # Variables from prior blocks (continuation snippets)
    "study.ask(",
    "study.tell(",
    "study.top_k(",
    "study.run(",
    "study.trial_count(",
    "study.pareto_front(",
    "study.trials(",
    "study.update_objectives(",
    "study.save(",
    # Pseudo-code with type annotations (not valid Python)
    "-> Trial",
    "-> CompletedTrial",
    "-> list[",
    "-> int",
    "-> Study",
]

# Patterns that indicate a block is pseudo-code, not real Python syntax.
# These blocks skip compilation entirely.
PSEUDO_CODE_MARKERS = [
    "-> Trial ",
    "-> CompletedTrial",
    "-> list[CompletedTrial]",
    "-> int\n",
    "-> Study\n",
    ", ...)",  # ellipsis placeholder for omitted arguments
]


def _extract_python_blocks(md_path: Path) -> list[tuple[int, str]]:
    """Extract (line_number, code) tuples from a markdown file."""
    text = md_path.read_text()
    blocks = []
    for m in PYTHON_BLOCK_RE.finditer(text):
        lineno = text[: m.start()].count("\n") + 2  # +2 for ```python line
        blocks.append((lineno, m.group(1)))
    return blocks


def _classify_block(code: str) -> str:
    """Classify a code block as 'exec', 'syntax', or 'skip'."""
    # Pseudo-code blocks that aren't valid Python at all
    if any(marker in code for marker in PSEUDO_CODE_MARKERS):
        return "skip"
    # Blocks referencing undefined names or needing external state
    if any(marker in code for marker in SYNTAX_ONLY_MARKERS):
        return "syntax"
    return "exec"


def _collect_doc_blocks():
    """Yield pytest params for all Python code blocks in docs."""
    markdown_files = [ROOT_DIR / "README.md", *sorted(DOCS_DIR.glob("*.md"))]
    for md_file in markdown_files:
        for lineno, code in _extract_python_blocks(md_file):
            mode = _classify_block(code)
            # README blocks are self-contained by design. Execute the local
            # quick start end-to-end; only the remote example needs a server.
            if md_file.name == "README.md" and "Study.connect(" not in code:
                mode = "exec"
            display_name = md_file.relative_to(ROOT_DIR).as_posix()
            yield pytest.param(
                display_name,
                lineno,
                code,
                mode,
                id=f"{display_name}:L{lineno}",
            )


_BLOCKS = list(_collect_doc_blocks())


def _make_hola_namespace() -> dict:
    """Build a namespace with all hola exports pre-imported."""
    import hola_opt

    ns: dict = {}
    for name in hola_opt.__all__:
        ns[name] = getattr(hola_opt, name)
    return ns


@pytest.mark.doctest_md
@pytest.mark.parametrize("filename,lineno,code,mode", _BLOCKS)
def test_doc_code_block(filename, lineno, code, mode):
    """Verify documentation code blocks compile and (where possible) execute."""
    if mode == "skip":
        pytest.skip("pseudo-code block")

    # Always check syntax for non-skip blocks
    try:
        compiled = compile(code, f"{filename}:{lineno}", "exec")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {filename} line {lineno}: {e}")

    if mode == "exec":
        # Pre-import hola names so snippet blocks that assume prior imports work
        ns = _make_hola_namespace()
        try:
            exec(compiled, ns)
        except ImportError as e:
            pytest.fail(f"Import error in {filename} line {lineno}: {e}")
        except Exception as e:
            pytest.fail(f"Runtime error in {filename} line {lineno}: {e}")


@pytest.mark.doctest_md
@pytest.mark.parametrize("filename", ["README.md", "CONTRIBUTING.md"])
def test_root_document_local_links_exist(filename):
    """Keep first-run and contributor links anchored to real repository files."""
    document = ROOT_DIR / filename
    for target in LOCAL_LINK_RE.findall(document.read_text()):
        if "://" in target or target.startswith("mailto:"):
            continue
        resolved = (document.parent / target).resolve()
        assert resolved.exists(), f"{filename} links to missing path {target!r}"


@pytest.mark.doctest_md
def test_root_commands_use_the_current_cli_and_python_layout():
    """Reject the stale names and flags that originally broke the quick start."""
    text = "\n".join(
        (ROOT_DIR / filename).read_text() for filename in ("README.md", "CONTRIBUTING.md")
    )
    for stale in ("robopt-cli", "robopt-py", "--command"):
        assert stale not in text
    assert "hola serve" in text
    assert "hola worker" in text
    assert "--exec" in text
    assert "--directory hola-py" in text
