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

"""Verify the metadata and provenance files shipped in HOLA packages."""

from __future__ import annotations

import email
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARGO_PACKAGES = ("opt_engine", "hola", "hola-cli", "hola-py")
REQUIRED_CARGO_METADATA = (
    "authors",
    "description",
    "documentation",
    "homepage",
    "license",
    "readme",
    "repository",
)
MAX_WHEEL_BYTES = 64 * 1024 * 1024
MAX_SDIST_BYTES = 5 * 1024 * 1024
DASHBOARD_FILES = ("index.html", "styles.css", "app.js")


def verify_cargo_packages() -> None:
    """Check registry metadata and the files Cargo puts in each crate."""
    raw = subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        text=True,
    )
    packages = {package["name"]: package for package in json.loads(raw)["packages"]}

    for name in CARGO_PACKAGES:
        package = packages[name]
        missing = [field for field in REQUIRED_CARGO_METADATA if not package.get(field)]
        if missing:
            raise SystemExit(f"{name}: missing Cargo metadata: {', '.join(missing)}")
        if package["license"] != "Apache-2.0":
            raise SystemExit(f"{name}: expected Apache-2.0, got {package['license']!r}")

        listing = subprocess.check_output(
            ["cargo", "package", "--locked", "--allow-dirty", "--list", "-p", name],
            cwd=ROOT,
            text=True,
        ).splitlines()
        for required in ("LICENSE-APACHE", "README.md"):
            if required not in listing:
                raise SystemExit(f"{name}: Cargo package omits {required}")


def parse_metadata(raw: bytes, artifact: Path) -> email.message.Message:
    metadata = email.message_from_bytes(raw)
    required = {
        "Summary": "package summary",
        "License-Expression": "SPDX license expression",
        "License-File": "bundled license declaration",
        "Project-URL": "project links",
    }
    for field, description in required.items():
        if not metadata.get_all(field):
            raise SystemExit(f"{artifact.name}: missing {description} ({field})")
    if metadata["License-Expression"] != "Apache-2.0":
        raise SystemExit(f"{artifact.name}: unexpected license expression")
    if not metadata.get_payload().strip():
        raise SystemExit(f"{artifact.name}: README/long description is empty")
    return metadata


def verify_wheel(path: Path) -> None:
    if path.stat().st_size > MAX_WHEEL_BYTES:
        raise SystemExit(
            f"{path.name}: wheel exceeds {MAX_WHEEL_BYTES // (1024 * 1024)} MiB budget"
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_paths = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise SystemExit(f"{path.name}: expected exactly one METADATA file")
        parse_metadata(archive.read(metadata_paths[0]), path)
        if not any(
            name.endswith(".dist-info/licenses/LICENSE-APACHE") for name in names
        ):
            raise SystemExit(f"{path.name}: wheel omits LICENSE-APACHE")
        for dashboard_file in DASHBOARD_FILES:
            expected = f"hola_opt/dashboard/{dashboard_file}"
            if expected not in names:
                raise SystemExit(f"{path.name}: wheel omits {expected}")
        unwanted = [
            name for name in names if "__pycache__" in name or name.endswith(".pyc")
        ]
        if unwanted:
            raise SystemExit(f"{path.name}: wheel contains cache artifacts: {unwanted}")


def verify_sdist(path: Path) -> None:
    if path.stat().st_size > MAX_SDIST_BYTES:
        raise SystemExit(
            f"{path.name}: sdist exceeds {MAX_SDIST_BYTES // (1024 * 1024)} MiB budget"
        )
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        for required in ("LICENSE-APACHE", "README.md", "pyproject.toml"):
            if not any(name.endswith(f"/{required}") for name in names):
                raise SystemExit(f"{path.name}: source distribution omits {required}")
        for dashboard_file in DASHBOARD_FILES:
            expected = f"/hola_opt/dashboard/{dashboard_file}"
            if not any(name.endswith(expected) for name in names):
                raise SystemExit(f"{path.name}: source distribution omits {expected[1:]}")
        pkg_info = [name for name in names if name.endswith("/PKG-INFO")]
        if len(pkg_info) != 1:
            raise SystemExit(f"{path.name}: expected exactly one PKG-INFO file")
        extracted = archive.extractfile(pkg_info[0])
        if extracted is None:
            raise SystemExit(f"{path.name}: cannot read PKG-INFO")
        parse_metadata(extracted.read(), path)


def verify_python_distributions(directory: Path) -> None:
    for dashboard_file in DASHBOARD_FILES:
        canonical = ROOT / "dashboard" / dashboard_file
        bundled = ROOT / "hola-py" / "hola_opt" / "dashboard" / dashboard_file
        if canonical.read_bytes() != bundled.read_bytes():
            raise SystemExit(
                f"bundled dashboard {dashboard_file} differs from canonical asset"
            )
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if not wheels or not sdists:
        raise SystemExit(
            f"{directory}: expected at least one wheel and one source distribution"
        )
    for wheel in wheels:
        verify_wheel(wheel)
    for sdist in sdists:
        verify_sdist(sdist)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "cargo":
        verify_cargo_packages()
        return
    if len(sys.argv) == 3 and sys.argv[1] == "python":
        verify_python_distributions(Path(sys.argv[2]))
        return
    raise SystemExit(f"usage: {sys.argv[0]} cargo | python DIST_DIRECTORY")


if __name__ == "__main__":
    main()
