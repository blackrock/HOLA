# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Test that Python code blocks in documentation compile and (where possible) execute."""

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)

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
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        for lineno, code in _extract_python_blocks(md_file):
            mode = _classify_block(code)
            yield pytest.param(
                md_file.name,
                lineno,
                code,
                mode,
                id=f"{md_file.stem}:L{lineno}",
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
        compiled = compile(code, f"docs/{filename}:{lineno}", "exec")
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
