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

"""Test CLI artifact staging without building or executing the native program."""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from stage_cli_for_tests import stage_cli


class StageCliTests(unittest.TestCase):
    def test_native_tar_and_windows_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix, name in (
                ("tar.gz", "hola-linux-x86_64"),
                ("zip", "hola-windows-x86_64.exe"),
            ):
                with self.subTest(suffix=suffix):
                    archive = root / f"archive.{suffix}"
                    if suffix == "zip":
                        with zipfile.ZipFile(archive, "w") as packed:
                            packed.writestr(name, b"packaged CLI")
                    else:
                        with tarfile.open(archive, "w:gz") as packed:
                            item = tarfile.TarInfo(name)
                            item.size = len(b"packaged CLI")
                            packed.addfile(item, io.BytesIO(b"packaged CLI"))
                    destination = root / "staged" / name
                    self.assertEqual(
                        stage_cli(archive, destination), destination.resolve()
                    )
                    self.assertEqual(destination.read_bytes(), b"packaged CLI")
                    if os.name != "nt":
                        self.assertTrue(os.access(destination, os.X_OK))

    def test_rejects_wrong_name_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cli.tar.gz"
            with tarfile.open(archive, "w:gz") as packed:
                item = tarfile.TarInfo("hola-linux-x86_64")
                item.type = tarfile.SYMTYPE
                item.linkname = "../somewhere"
                packed.addfile(item)
            for name in ("hola-linux-x86_64", "wrong-name"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    stage_cli(archive, root / "staged" / name)


if __name__ == "__main__":
    unittest.main()
