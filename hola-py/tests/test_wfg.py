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

"""Correctness tests for the WFG functions' pymoo delegation and rescaling."""

import sys
from pathlib import Path

import pytest

HOLA_PY_DIR = Path(__file__).parent.parent
if str(HOLA_PY_DIR) not in sys.path:
    sys.path.insert(0, str(HOLA_PY_DIR))

pytest.importorskip("pymoo")

import numpy as np  # noqa: E402
from pymoo.problems import get_problem  # noqa: E402

from benchmarks.functions import wfg  # noqa: E402
from benchmarks.functions.wfg import _extract_vec  # noqa: E402

WFG_FUNCS = {"wfg1": wfg.wfg1, "wfg4": wfg.wfg4, "wfg9": wfg.wfg9}


def test_extract_vec_numeric_order():
    """_extract_vec orders x-keys by numeric suffix, not lexicographically.

    With keys x0..x11 a lexicographic sort places x10/x11 before x2, so this
    asserts the decision vector comes out in index order.
    """
    n = 12
    shuffled = [10, 0, 3, 11, 7, 1, 5, 2, 9, 4, 8, 6]
    p = {f"x{i}": float(i) for i in shuffled}
    vec = _extract_vec(p)
    assert np.array_equal(vec, np.arange(n, dtype=float))


@pytest.mark.benchmarks
@pytest.mark.parametrize("name", list(WFG_FUNCS))
@pytest.mark.parametrize("n_obj", [2, 3])
def test_wfg_matches_pymoo(name, n_obj):
    """Each WFG function matches pymoo's get_problem evaluated on rescaled inputs."""
    func = WFG_FUNCS[name]
    n_var = (n_obj - 1) + 4
    scale = np.array([2.0 * (i + 1) for i in range(n_var)])
    prob = get_problem(name, n_var=n_var, n_obj=n_obj)

    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.random(n_var)
        params = {f"x{i}": float(x[i]) for i in range(n_var)}

        result = func(params, n_obj=n_obj)
        assert len(result) == n_obj
        got = np.array([result[f"f{m + 1}"] for m in range(n_obj)])
        assert np.all(np.isfinite(got))

        expected = prob.evaluate((x * scale).reshape(1, -1))[0]
        assert np.allclose(got, expected, atol=0.0, rtol=0.0)
