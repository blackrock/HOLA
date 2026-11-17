#!/usr/bin/env python3
"""Reject install instructions that expose hola-opt to package-index confusion.

``hola-opt`` is distributed from this repository's own package index. An
installer told to consult a second index treats both as equally authoritative,
so whoever registers the same name on the other index can have their upload
installed instead. Neither index order, ``--pre``, nor an exact version pin
prevents that. This check keeps such instructions out of the repository:

1. No command, code block, or configuration file may combine a private package
   name with an additional package index (``--extra-index-url``,
   ``PIP_EXTRA_INDEX_URL``, ``extra-index-url =``, uv's unsafe index
   strategies, ...).
2. While the public index is not authoritative for a private package, every
   command that installs it by name must use only a trusted index, no index at
   all, or a hash-verified artifact.
3. The user-facing install command is identical in every synced document.
4. The hash-locked install file pins every requirement to a version and to
   SHA-256 digests, and resolves from the trusted index only.

Run from anywhere: ``python3 .github/scripts/check_install_docs.py``. Exits
nonzero and lists each violation as ``path:line: message``. Stdlib-only.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

# Distribution names that must never be resolved from an index we do not
# control. Compared after PEP 503 normalization.
PRIVATE_PACKAGES = ("hola-opt",)

# Set to True only once the PyPI project of every private package is owned by
# the maintainers and carries the official release artifacts. Until then a
# plain "pip install <name>" asks PyPI for a name anyone can still register.
PUBLIC_INDEX_IS_AUTHORITATIVE = False

TRUSTED_INDEX_URLS = ("https://blackrock.github.io/HOLA/simple/",)

LOCK_FILE = "install/hola-opt-requirements.txt"

# The one documented way to install the published package. Change it here and
# in every synced document in the same commit.
CANONICAL_INSTALL = (
    "pip install --require-hashes -r "
    "https://raw.githubusercontent.com/blackrock/HOLA/main/" + LOCK_FILE
)
SYNCED_DOCS = ("README.md", "docs/getting-started.md", "docs/index.md")

MARKDOWN_SUFFIXES = {".md", ".markdown"}
# Files that are a single installer configuration: an extra index anywhere in
# the file applies to every requirement in it.
WHOLE_FILE_SUFFIXES = {".txt", ".in", ".cfg", ".ini", ".conf", ".toml"}
LINE_SUFFIXES = {".yml", ".yaml", ".sh", ".bash", ".ps1", ".bat", ".py", ".json"}
SKIPPED_NAMES = {"Cargo.lock", "uv.lock", "package-lock.json"}
# These files quote unsafe commands on purpose.
SELF_FILES = {
    ".github/scripts/check_install_docs.py",
    ".github/scripts/test_check_install_docs.py",
}

EXTRA_INDEX_RE = re.compile(
    r"extra[-_]index(?:[-_]urls?)?\b|unsafe-(?:first|best)-match|unsafe-any-match",
    re.IGNORECASE,
)
INSTALLER_RE = re.compile(
    r"(?<![\w./-])(?:pip[0-9.]*|pipx|uv|poetry|pdm|conda|mamba|easy_install)(?![\w/-])",
    re.IGNORECASE,
)
INSTALL_VERBS = {"install", "add", "download", "wheel"}
# Options whose following token is a value, not a requirement.
VALUE_OPTIONS = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-e",
    "--editable",
    "-i",
    "--index-url",
    "--default-index",
    "--index",
    "--extra-index-url",
    "-f",
    "--find-links",
    "-d",
    "--dest",
    "-t",
    "--target",
    "--python",
    "-p",
    "--prefix",
    "--root",
    "--platform",
    "--python-version",
    "--abi",
    "--implementation",
    "--only-binary",
    "--no-binary",
    "--cache-dir",
    "--directory",
    "--group",
    "--extra",
    "--with",
    "--from",
}
INDEX_OPTIONS = {"-i", "--index-url", "--default-index"}
HASHED_DIRECT_URL_RE = re.compile(r"@\s*\S+#sha256=[0-9a-f]{64}\b")
LOCK_HASH_RE = re.compile(r"--hash=sha256:[0-9a-f]{64}(?![0-9a-f])")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")


def normalize(name: str) -> str:
    """PEP 503 project-name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


_PRIVATE = {normalize(name) for name in PRIVATE_PACKAGES}
_PRIVATE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(re.escape(name).replace(r"\-", "[-_.]+") for name in sorted(_PRIVATE))
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def mentions_private(text: str) -> bool:
    return _PRIVATE_RE.search(text) is not None


def requirement_name(token: str) -> str | None:
    """Return the normalized project name if ``token`` is a named requirement.

    Paths and bare URLs (``dist/hola_opt-1.0.whl``, ``https://...``) are not
    named requirements: they install exactly that artifact.
    """
    token = token.strip("\"'`()[]{},;")
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$", token)
    if match is None:
        return None
    name, rest = match.group(1), match.group(2)
    if rest and rest[0] not in "=<>!~@;":
        return None
    if not rest and re.search(r"\.(whl|tar\.gz|zip|txt|toml|cfg|ini)$", name, re.IGNORECASE):
        return None
    return normalize(name.rstrip("."))


def split_command(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return line.split()


def unsafe_named_installs(line: str) -> list[str]:
    """Return the private packages ``line`` installs by name without a safe source."""
    if INSTALLER_RE.search(line) is None:
        return []
    tokens = split_command(line)
    verb_at = next((i for i, tok in enumerate(tokens) if tok.lower() in INSTALL_VERBS), None)
    if verb_at is None or not any(INSTALLER_RE.search(tok) for tok in tokens[:verb_at]):
        return []

    named: list[tuple[str, str]] = []
    options: set[str] = set()
    index_values: list[str] = []
    args = tokens[verb_at + 1 :]
    skip = False
    for position, token in enumerate(args):
        if skip:
            skip = False
            continue
        if token in ("&&", "||", "|", ";"):
            break
        if token.startswith("-"):
            option, _, inline_value = token.partition("=")
            options.add(option)
            if option in INDEX_OPTIONS:
                index_values.append(
                    inline_value or (args[position + 1] if position + 1 < len(args) else "")
                )
            if option in VALUE_OPTIONS and not inline_value:
                skip = True
            continue
        name = requirement_name(token)
        if name in _PRIVATE:
            named.append((name, token))
    if not named:
        return []

    if "--no-index" in options or "--require-hashes" in options:
        return []
    if index_values and all(
        value.startswith("file:") or value in TRUSTED_INDEX_URLS for value in index_values
    ):
        return []
    return [name for name, token in named if HASHED_DIRECT_URL_RE.search(token) is None]


def logical_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Join backslash (and PowerShell backtick) continuations."""
    joined: list[tuple[int, str]] = []
    start, parts = 0, []
    for number, text in lines:
        if not parts:
            start = number
        stripped = text.rstrip()
        if stripped.endswith(("\\", " `")):
            parts.append(stripped[:-1])
            continue
        parts.append(stripped)
        joined.append((start, " ".join(part.strip() for part in parts)))
        parts = []
    if parts:
        joined.append((start, " ".join(part.strip() for part in parts)))
    return joined


def markdown_units(text: str):
    """Yield ``(kind, line, text)``: fenced blocks and inline code spans."""
    fence = None
    block: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = FENCE_RE.match(line)
        if fence is None:
            if match:
                fence, block = match.group(1), []
                continue
            for span in INLINE_CODE_RE.finditer(line):
                yield "inline", number, span.group(2)
        elif match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
            yield "block", block[0][0] if block else number, block
            fence = None
        else:
            block.append((number, line))
    if fence is not None and block:
        yield "block", block[0][0], block


def check_text(path: str, text: str) -> list[str]:
    """Return violations of rules 1 and 2 for one file."""
    problems: list[str] = []

    def extra_index(line: int, unit: str) -> None:
        if EXTRA_INDEX_RE.search(unit) and mentions_private(unit):
            problems.append(
                f"{path}:{line}: additional package index used with a private package; "
                "resolve it from the trusted index only (--index-url) or from a "
                "hash-locked requirements file"
            )

    def named_install(line: int, command: str) -> None:
        if PUBLIC_INDEX_IS_AUTHORITATIVE:
            return
        for name in unsafe_named_installs(command):
            problems.append(
                f"{path}:{line}: installs {name} by name from an index that is not "
                "authoritative for it; use the hash-locked install command"
            )

    suffix = Path(path).suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        for kind, line, unit in markdown_units(text):
            if kind == "inline":
                extra_index(line, unit)
                named_install(line, unit)
                continue
            extra_index(line, "\n".join(body for _, body in unit))
            for number, command in logical_lines(unit):
                named_install(number, command)
        return problems

    numbered = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if not line.lstrip().startswith("#")
    ]
    if suffix in WHOLE_FILE_SUFFIXES:
        body = "\n".join(line for _, line in numbered)
        first = next((n for n, line in numbered if EXTRA_INDEX_RE.search(line)), 1)
        extra_index(first, body)
    for number, command in logical_lines(numbered):
        if suffix not in WHOLE_FILE_SUFFIXES:
            extra_index(number, command)
        named_install(number, command)
    return problems


def check_synced_docs(documents: dict[str, str]) -> list[str]:
    """Rule 3: every synced document carries the canonical install command."""
    problems = []
    for path in SYNCED_DOCS:
        text = documents.get(path)
        if text is None:
            problems.append(f"{path}:1: synced install document is missing")
            continue
        commands = {
            command.strip()
            for kind, _, unit in markdown_units(text)
            if kind == "block"
            for _, command in logical_lines(unit)
        }
        if CANONICAL_INSTALL not in commands:
            problems.append(
                f"{path}:1: install command differs from the canonical one; "
                f"expected a code block line reading: {CANONICAL_INSTALL}"
            )
    return problems


def check_lock_file(path: str, text: str) -> list[str]:
    """Rule 4: the install file is fully hash-locked to the trusted index."""
    problems = []
    numbered = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    index_urls = []
    requirements = 0
    for number, entry in logical_lines(numbered):
        if entry.startswith("-"):
            option, _, value = entry.replace("=", " ", 1).partition(" ")
            if option in INDEX_OPTIONS:
                index_urls.append(value.strip())
            elif option in ("-f", "--find-links") or EXTRA_INDEX_RE.search(option):
                problems.append(f"{path}:{number}: {option} adds a second package source")
            continue
        requirements += 1
        spec = entry.split("--hash", 1)[0].strip()
        name = requirement_name(spec.split("==", 1)[0])
        if name not in _PRIVATE:
            problems.append(f"{path}:{number}: unexpected requirement {spec!r}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+==[A-Za-z0-9.!+_-]+", spec):
            problems.append(f"{path}:{number}: {spec!r} is not pinned with ==")
        declared = entry.count("--hash")
        if declared == 0 or declared != len(LOCK_HASH_RE.findall(entry)):
            problems.append(f"{path}:{number}: every --hash must be a full sha256 digest")
    if requirements == 0:
        problems.append(f"{path}:1: no requirements found")
    if len(index_urls) != 1 or index_urls[0] not in TRUSTED_INDEX_URLS:
        problems.append(
            f"{path}:1: must set exactly one --index-url from {list(TRUSTED_INDEX_URLS)}; "
            f"found {index_urls}"
        )
    return problems


def tracked_files(root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.decode("utf-8", "surrogateescape")
    return [name for name in output.split("\0") if name]


def check_repository(root: Path) -> list[str]:
    problems: list[str] = []
    documents: dict[str, str] = {}
    scanned = MARKDOWN_SUFFIXES | WHOLE_FILE_SUFFIXES | LINE_SUFFIXES
    for path in tracked_files(root):
        file = Path(path)
        if file.suffix.lower() not in scanned or file.name in SKIPPED_NAMES:
            continue
        if path in SELF_FILES or not (root / path).is_file():
            continue
        try:
            text = (root / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path in SYNCED_DOCS:
            documents[path] = text
        problems.extend(check_text(path, text))
        if path == LOCK_FILE:
            problems.extend(check_lock_file(path, text))
    problems.extend(check_synced_docs(documents))
    if not (root / LOCK_FILE).is_file():
        problems.append(f"{LOCK_FILE}:1: hash-locked install file is missing")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    problems = check_repository(root)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} unsafe install instruction(s) found", file=sys.stderr)
        return 1
    print("install instructions OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
