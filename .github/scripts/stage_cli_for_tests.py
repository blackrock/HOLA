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

"""Extract one expected native CLI from a release archive for integration tests."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def stage_cli(archive: Path, destination: Path) -> Path:
    """Extract exactly the expected binary, preserving no archive-selected paths."""
    name = destination.name
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as packed:
            entries = [item for item in packed.infolist() if item.filename == name]
            if len(entries) != 1 or entries[0].is_dir():
                raise ValueError(f"{archive}: expected exactly one binary named {name}")
            payload = packed.read(entries[0])
    else:
        with tarfile.open(archive, "r:gz") as packed:
            entries = [item for item in packed.getmembers() if item.name == name]
            if len(entries) != 1 or not entries[0].isfile():
                raise ValueError(
                    f"{archive}: expected exactly one regular binary named {name}"
                )
            extracted = packed.extractfile(entries[0])
            if extracted is None:
                raise ValueError(f"{archive}: cannot read {name}")
            with extracted:
                payload = extracted.read()
    if not payload:
        raise ValueError(f"{archive}: CLI binary is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    destination.chmod(0o755)
    return destination.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(stage_cli(args.archive, args.destination))


if __name__ == "__main__":
    main()
