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

"""Check documented objective functions without requiring the Rust bindings."""

import ast
import math
import re
import runpy
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


class TestDocumentedObjectives(unittest.TestCase):
    def test_forrester_examples_match_benchmark(self):
        reference = runpy.run_path(
            str(ROOT_DIR / "hola-py/benchmarks/functions/single_objective.py")
        )["forrester"]
        for filename in ("README.md", "docs/getting-started.md"):
            functions: list[ast.stmt] = [
                node
                for block in PYTHON_BLOCK_RE.findall((ROOT_DIR / filename).read_text())
                for node in ast.parse(block).body
                if isinstance(node, ast.FunctionDef) and node.name == "forrester"
            ]
            self.assertEqual(len(functions), 1, filename)
            # Evaluate only the objective, not the study or remote-client examples.
            module = ast.Module(body=functions, type_ignores=[])
            namespace: dict = {"math": math}
            exec(compile(module, filename, "exec"), namespace)
            for x in (0.0, 0.25, 0.5, 0.757248757841856, 1.0):
                with self.subTest(filename=filename, x=x):
                    params = {"x": x}
                    self.assertAlmostEqual(
                        namespace["forrester"](params)["value"],
                        reference(params),
                        delta=1e-12,
                    )
