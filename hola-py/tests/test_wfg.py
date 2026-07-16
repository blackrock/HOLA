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

import hashlib
import subprocess
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
from benchmarks.functions.wfg import (  # noqa: E402
    WFG_REFERENCE_FRONT_VERSION,
    _extract_vec,
    _wfg_parameters,
    _wfg_reference_directions,
)
from benchmarks.problems.multi_objective import MULTI_OBJECTIVE_PROBLEMS  # noqa: E402

WFG_FUNCS = {"wfg1": wfg.wfg1, "wfg4": wfg.wfg4, "wfg9": wfg.wfg9}
WFG_FRONT_FUNCS = {
    "wfg1": wfg.wfg1_pareto_front,
    "wfg4": wfg.wfg4_pareto_front,
    "wfg9": wfg.wfg9_pareto_front,
}

EXPECTED_REFERENCE_IDENTITIES = {
    "wfg1_2obj_24d": (
        "896b3a97ca1c224d7c95b8371dc135320eaa504763d33939fdfffba0d6a46598",
        0.6855149849723752,
    ),
    "wfg4_2obj_24d": (
        "050ad81783c23b34cea705a41fa96c78384bd94e825457389c732f2e0eee566b",
        0.3501725676827616,
    ),
    "wfg9_2obj_24d": (
        "2a6b2318c9d4fdcd0e8c8bf923b1e98ac33ac60c6a08fe596783991267f9c011",
        0.35017256768276245,
    ),
    "wfg1_3obj_24d": (
        "648ffa9f576938708254425b3973697c499042be893695e2aacdbfe032af91b6",
        0.9544874155255598,
    ),
    "wfg4_3obj_24d": (
        "b1d7f4ed6fc6a76e3c52891dee41f3ce5d8eb6f3e8f1b71e61b09f8083e840f6",
        0.5872082028983905,
    ),
    "wfg9_3obj_24d": (
        "7718013773f512d2476173272940754747c3b7aedc1fa4e201e2437c48476daf",
        0.5872082028983898,
    ),
}


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
    k, distance_dimensions = _wfg_parameters(n_obj)
    n_var = k + distance_dimensions
    scale = np.array([2.0 * (i + 1) for i in range(n_var)])
    prob = get_problem(name, n_var=n_var, n_obj=n_obj, k=k, l=distance_dimensions)
    assert prob.k == 4
    assert prob.l == 20

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


@pytest.mark.parametrize("name", list(WFG_FUNCS))
def test_wfg_rejects_dimensions_inconsistent_with_standard_k_and_l(name):
    with pytest.raises(ValueError, match=r"requires k=4, l=20.*24 variables; got 25"):
        WFG_FUNCS[name]({f"x{i}": 0.5 for i in range(25)}, n_obj=2)


@pytest.mark.parametrize("n_obj", [2, 3])
def test_registered_wfg_problems_use_standard_dimensions(n_obj):
    for variant in WFG_FUNCS:
        problem = next(
            problem
            for problem in MULTI_OBJECTIVE_PROBLEMS.values()
            if problem.family == variant and problem.n_objectives == n_obj
        )
        assert problem.dimensionality == 24


@pytest.mark.benchmarks
@pytest.mark.parametrize("n_obj, expected_points", [(2, 500), (3, 496)])
def test_wfg_reference_directions_have_documented_deterministic_density(n_obj, expected_points):
    first = _wfg_reference_directions(n_obj, 500)
    second = _wfg_reference_directions(n_obj, 500)

    assert first.shape == (expected_points, n_obj)
    assert np.array_equal(first, second)
    assert np.all(first >= 0.0)
    assert np.allclose(first.sum(axis=1), 1.0, atol=1e-15, rtol=0.0)
    assert len(np.unique(first, axis=0)) == expected_points


@pytest.mark.benchmarks
@pytest.mark.parametrize("name", list(WFG_FRONT_FUNCS))
@pytest.mark.parametrize("n_obj, expected_points", [(2, 500), (3, 496)])
def test_wfg_reference_front_is_direction_matched_and_rng_independent(name, n_obj, expected_points):
    directions = _wfg_reference_directions(n_obj, 500)
    np.random.seed(1)
    first = WFG_FRONT_FUNCS[name](n_obj)
    np.random.seed(987654321)
    second = WFG_FRONT_FUNCS[name](n_obj)

    assert first.shape == (expected_points, n_obj)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert len(np.unique(first, axis=0)) == expected_points

    # WFG1's optimized evaluator has a small common floating-point distance
    # offset.  Removing each coordinate's minimum and scale exposes the exact
    # objective-space directions for all three variants.
    shifted = first - first.min(axis=0)
    scaled = shifted / shifted.max(axis=0)
    front_directions = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    expected_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    assert np.allclose(front_directions, expected_directions, atol=1e-12, rtol=0.0)


@pytest.mark.benchmarks
def test_registered_wfg_reference_identity_and_hypervolume_are_stable():
    assert WFG_REFERENCE_FRONT_VERSION == "das-dennis-direction-mapped-v1"

    for name, (expected_digest, expected_hv) in EXPECTED_REFERENCE_IDENTITIES.items():
        problem = MULTI_OBJECTIVE_PROBLEMS[name]
        assert problem.true_pareto_front is not None
        canonical_front = np.ascontiguousarray(
            np.round(np.asarray(problem.true_pareto_front), decimals=12).astype("<f8")
        )
        digest = hashlib.sha256(canonical_front.tobytes()).hexdigest()
        assert digest == expected_digest
        assert problem.normalized_reference_hypervolume == pytest.approx(
            expected_hv, abs=2e-14, rel=0.0
        )


@pytest.mark.benchmarks
@pytest.mark.parametrize("n_obj", [2, 3])
def test_wfg1_ideal_is_the_implemented_front_coordinatewise_minimum(n_obj):
    problem = next(
        problem
        for problem in MULTI_OBJECTIVE_PROBLEMS.values()
        if problem.family == "wfg1" and problem.n_objectives == n_obj
    )
    assert problem.true_pareto_front is not None
    assert problem.ideal_point == pytest.approx(problem.true_pareto_front.min(axis=0))
    assert problem.ideal_point == pytest.approx(
        (0.0694639534683812,) * n_obj,
        abs=2e-16,
        rel=0.0,
    )


@pytest.mark.benchmarks
def test_wfg_references_and_hypervolumes_are_byte_identical_across_fresh_processes():
    script = """
import hashlib
from benchmarks.problems.multi_objective import MULTI_OBJECTIVE_PROBLEMS

for name, problem in MULTI_OBJECTIVE_PROBLEMS.items():
    if problem.family.startswith("wfg"):
        digest = hashlib.sha256(problem.true_pareto_front.tobytes()).hexdigest()
        print(name, digest, problem.normalized_reference_hypervolume.hex())
"""
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=HOLA_PY_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
