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

"""Immutable provenance manifests for benchmark campaigns."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "6"
MANIFEST_FILENAME = "campaign_manifest.json"

_NATIVE_EXTENSION_MODULE = "hola_opt.hola_opt"

_DEPENDENCIES = (
    "hola-opt",
    "numpy",
    "optuna",
    "pandas",
    "pymoo",
    "scikit-learn",
    "scipy",
)
_SOURCE_PREFIXES = (
    "opt_engine/",
    "hola/",
    "hola-py/benchmarks/",
    "hola-py/hola_opt/",
    "hola-py/src/",
)
_SOURCE_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "hola-py/pyproject.toml",
    "hola-py/uv.lock",
    "rust-toolchain",
    "rust-toolchain.toml",
}
_SOURCE_SUFFIXES = {".py", ".rs"}
_LOCK_FILES = (
    "Cargo.lock",
    "hola-py/uv.lock",
    "rust-toolchain",
    "rust-toolchain.toml",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_fingerprint(payload: dict[str, Any]) -> str:
    """Hash a manifest payload, excluding any existing fingerprint field."""
    fingerprint_payload = {key: value for key, value in payload.items() if key != "fingerprint"}
    return hashlib.sha256(_canonical_json(fingerprint_payload).encode()).hexdigest()


def attach_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with its deterministic fingerprint."""
    manifest = dict(payload)
    manifest["fingerprint"] = manifest_fingerprint(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject a malformed or manually altered manifest."""
    actual = manifest.get("fingerprint")
    expected = manifest_fingerprint(manifest)
    if not isinstance(actual, str) or actual != expected:
        raise RuntimeError("campaign manifest fingerprint is missing or invalid")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("campaign manifest is missing provenance")
    _validate_provenance(provenance)


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _source_paths(repo_root: Path) -> list[Path]:
    listed = _git_output(repo_root, "ls-files", "--cached", "--others", "--exclude-standard")
    if listed is not None:
        relative_paths = [Path(line) for line in listed.splitlines() if line]
    else:
        relative_paths = []
        for prefix in _SOURCE_PREFIXES:
            root = repo_root / prefix
            if root.exists():
                relative_paths.extend(
                    path.relative_to(repo_root) for path in root.rglob("*") if path.is_file()
                )
        relative_paths.extend(
            path.relative_to(repo_root)
            for path in repo_root.rglob("Cargo.toml")
            if "target" not in path.relative_to(repo_root).parts
        )
        relative_paths.extend(Path(path) for path in _SOURCE_FILES)

    selected = {
        path
        for path in relative_paths
        if path.as_posix() in _SOURCE_FILES
        or (path.name == "Cargo.toml" and "target" not in path.parts)
        or (
            path.suffix in _SOURCE_SUFFIXES
            and any(path.as_posix().startswith(prefix) for prefix in _SOURCE_PREFIXES)
        )
    }
    return sorted(selected, key=lambda path: path.as_posix())


def _hash_files(repo_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        path = repo_root / relative
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dependency in _DEPENDENCIES:
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    return versions


def _native_extension_identity(
    installed_version: str | None,
    *,
    extension_path: Path | None = None,
    module_name: str = _NATIVE_EXTENSION_MODULE,
) -> dict[str, Any] | None:
    """Identify the loaded HOLA extension without recording a machine path.

    ``extension_path`` is a test seam. Production collection imports
    ``hola_opt.hola_opt`` and hashes the exact shared object named by the loaded
    module.
    """
    if installed_version is None:
        return None

    if extension_path is None:
        try:
            native_module = importlib.import_module(module_name)
        except (ImportError, OSError) as error:
            raise RuntimeError(
                f"hola-opt {installed_version} is installed, but its native extension "
                f"module {module_name!r} could not be loaded; rebuild the active environment "
                "with `maturin develop` and rerun the campaign"
            ) from error
        module_name = native_module.__name__
        location = getattr(native_module, "__file__", None)
        if not isinstance(location, str) or not location:
            raise RuntimeError(
                f"hola-opt {installed_version} is installed, but loaded module "
                f"{module_name!r} does not identify its native extension file; rebuild the "
                "active environment with `maturin develop` and rerun the campaign"
            )
        extension_path = Path(location)

    filename = extension_path.name
    if not filename.startswith("hola_opt") or extension_path.suffix != ".so":
        raise RuntimeError(
            f"hola-opt {installed_version} is installed, but loaded module {module_name!r} "
            f"does not point to a hola_opt*.so native extension (reported {filename!r}); "
            "rebuild the active environment with `maturin develop` and rerun the campaign"
        )

    digest = hashlib.sha256()
    byte_size = 0
    try:
        with extension_path.open("rb") as extension:
            while chunk := extension.read(1024 * 1024):
                digest.update(chunk)
                byte_size += len(chunk)
    except OSError as error:
        raise RuntimeError(
            f"hola-opt {installed_version} is installed, but the native extension for loaded "
            f"module {module_name!r} ({filename}) cannot be read; repair or rebuild the active "
            "environment with `maturin develop` and rerun the campaign"
        ) from error

    return {
        "module": module_name,
        "filename": filename,
        "byte_size": byte_size,
        "sha256": digest.hexdigest(),
    }


def _validate_provenance(provenance: dict[str, Any]) -> None:
    """Require binary identity whenever provenance reports hola-opt installed."""
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, dict) or "hola-opt" not in dependencies:
        raise RuntimeError("campaign provenance must report the hola-opt dependency version")

    installed_version = dependencies["hola-opt"]
    native_extension = provenance.get("native_extension")
    if installed_version is None:
        if native_extension is not None:
            raise RuntimeError(
                "campaign provenance reports a native extension while hola-opt is not installed"
            )
        return
    if not isinstance(native_extension, dict):
        raise RuntimeError(
            "hola-opt is reported installed, but campaign provenance has no native-extension "
            "identity; rebuild the active environment with `maturin develop` and rerun the "
            "campaign"
        )

    required = {"module", "filename", "byte_size", "sha256"}
    if set(native_extension) != required:
        raise RuntimeError(
            "hola-opt native-extension provenance must contain module, filename, byte_size, "
            "and sha256"
        )
    module = native_extension["module"]
    filename = native_extension["filename"]
    byte_size = native_extension["byte_size"]
    sha256 = native_extension["sha256"]
    if not isinstance(module, str) or not module:
        raise RuntimeError("hola-opt native-extension provenance has an invalid module identity")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.startswith("hola_opt")
        or not filename.endswith(".so")
    ):
        raise RuntimeError("hola-opt native-extension provenance has an invalid filename")
    if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
        raise RuntimeError("hola-opt native-extension provenance has an invalid byte_size")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError("hola-opt native-extension provenance has an invalid sha256")


def collect_provenance(
    repo_root: Path | None = None,
    *,
    dependency_versions: dict[str, str | None] | None = None,
    native_extension_path: Path | None = None,
) -> dict[str, Any]:
    """Collect code, lockfile, interpreter, platform, and dependency identity."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    source_paths = _source_paths(repo_root)
    status = (
        _git_output(
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *(path.as_posix() for path in source_paths),
        )
        if source_paths
        else None
    )
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    dependencies = _dependency_versions() if dependency_versions is None else dependency_versions
    native_extension = _native_extension_identity(
        dependencies.get("hola-opt"), extension_path=native_extension_path
    )

    provenance = {
        "code": {
            "commit": commit,
            "dirty": None if status is None else bool(status),
            "source_hash": _hash_files(repo_root, source_paths),
        },
        "lock_hash": _hash_files(repo_root, [Path(path) for path in _LOCK_FILES]),
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "dependencies": dependencies,
        "native_extension": native_extension,
    }
    _validate_provenance(provenance)
    return provenance


def build_campaign_manifest(
    *,
    run_kind: str,
    budgets: list[int],
    n_runs: int,
    problem_names: list[str],
    optimizer_names: list[str],
    optimizer_configurations: list[dict[str, Any]],
    campaign_configuration: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic manifest expected for one result directory."""
    resolved_provenance = collect_provenance() if provenance is None else provenance
    _validate_provenance(resolved_provenance)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": run_kind,
        "budgets": list(budgets),
        "n_runs": n_runs,
        "problems": list(problem_names),
        "optimizers": list(optimizer_names),
        "optimizer_configurations": optimizer_configurations,
        "provenance": resolved_provenance,
    }
    if campaign_configuration is not None:
        payload["campaign_configuration"] = campaign_configuration
    return attach_fingerprint(payload)
