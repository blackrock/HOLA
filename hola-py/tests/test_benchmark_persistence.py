# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for benchmark provenance and audit-ready persistence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.data.manifest import (
    MANIFEST_FILENAME,
    attach_fingerprint,
    build_campaign_manifest,
    collect_provenance,
)
from benchmarks.data.persistence import ResultStore

pytestmark = pytest.mark.benchmarks


def _manifest(*, budgets: list[int] | None = None, run_kind: str = "single_objective") -> dict:
    return build_campaign_manifest(
        run_kind=run_kind,
        budgets=[10] if budgets is None else budgets,
        n_runs=2,
        problem_names=["quadratic"],
        optimizer_names=["optimizer"],
        optimizer_configurations=[
            {
                "optimizer": "optimizer",
                "by_budget": [
                    {
                        "budget": budget,
                        "configuration": {"adapter": "test", "budget": budget},
                    }
                    for budget in ([10] if budgets is None else budgets)
                ],
            }
        ],
        provenance={
            "code": {"commit": "abc123", "dirty": False, "source_hash": "source"},
            "lock_hash": "lock",
            "python": {"implementation": "CPython", "version": "3.test"},
            "platform": {"platform": "test", "machine": "test", "system": "test"},
            "dependencies": {"hola-opt": "test", "numpy": "test"},
            "native_extension": {
                "module": "hola_opt.hola_opt",
                "filename": "hola_opt.abi3.so",
                "byte_size": 4,
                "sha256": "0" * 64,
            },
        },
    )


def test_manifest_records_protocol_and_reproducibility_identity() -> None:
    manifest = _manifest()

    assert manifest["run_kind"] == "single_objective"
    assert manifest["budgets"] == [10]
    assert manifest["n_runs"] == 2
    assert manifest["problems"] == ["quadratic"]
    assert manifest["optimizers"] == ["optimizer"]
    assert manifest["optimizer_configurations"] == [
        {
            "optimizer": "optimizer",
            "by_budget": [{"budget": 10, "configuration": {"adapter": "test", "budget": 10}}],
        }
    ]
    assert manifest["protocol_version"]
    assert len(manifest["fingerprint"]) == 64
    assert set(manifest["provenance"]) == {
        "code",
        "lock_hash",
        "python",
        "platform",
        "dependencies",
        "native_extension",
    }


def test_collected_provenance_has_required_environment_fields() -> None:
    provenance = collect_provenance()

    assert set(provenance["code"]) == {"commit", "dirty", "source_hash"}
    assert len(provenance["code"]["source_hash"]) == 64
    assert len(provenance["lock_hash"]) == 64
    assert set(provenance["python"]) == {"implementation", "version"}
    assert set(provenance["platform"]) == {"platform", "machine", "system"}
    assert "numpy" in provenance["dependencies"]
    assert set(provenance["native_extension"]) == {
        "module",
        "filename",
        "byte_size",
        "sha256",
    }
    assert provenance["native_extension"]["module"] == "hola_opt.hola_opt"
    assert provenance["native_extension"]["filename"].startswith("hola_opt")
    assert "/" not in provenance["native_extension"]["filename"]
    assert len(provenance["native_extension"]["sha256"]) == 64


def _collect_with_test_extension(repo_root: Path, extension_path: Path, payload: bytes) -> dict:
    extension_path.write_bytes(payload)
    return collect_provenance(
        repo_root,
        dependency_versions={"hola-opt": "1.test"},
        native_extension_path=extension_path,
    )


def test_native_extension_bytes_are_part_of_manifest_identity(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "Cargo.toml").write_text("[workspace]\n")
    extension_path = tmp_path / "hola_opt.abi3.so"

    first = _collect_with_test_extension(repo_root, extension_path, b"native-one")
    second = _collect_with_test_extension(repo_root, extension_path, b"native-two")
    assert first["code"] == second["code"]
    assert first["lock_hash"] == second["lock_hash"]
    assert first["native_extension"] == {
        "module": "hola_opt.hola_opt",
        "filename": "hola_opt.abi3.so",
        "byte_size": 10,
        "sha256": hashlib.sha256(b"native-one").hexdigest(),
    }
    assert first["native_extension"]["sha256"] != second["native_extension"]["sha256"]

    first_manifest = build_campaign_manifest(
        run_kind="single_objective",
        budgets=[25],
        n_runs=1,
        problem_names=["problem"],
        optimizer_names=["optimizer"],
        optimizer_configurations=[],
        provenance=first,
    )
    second_manifest = build_campaign_manifest(
        run_kind="single_objective",
        budgets=[25],
        n_runs=1,
        problem_names=["problem"],
        optimizer_names=["optimizer"],
        optimizer_configurations=[],
        provenance=second,
    )
    assert first_manifest["fingerprint"] != second_manifest["fingerprint"]


def test_source_only_provenance_cannot_stand_in_for_installed_binary() -> None:
    provenance = {
        "code": {"commit": "abc123", "dirty": False, "source_hash": "source"},
        "lock_hash": "lock",
        "python": {"implementation": "CPython", "version": "test"},
        "platform": {"platform": "test", "machine": "test", "system": "test"},
        "dependencies": {"hola-opt": "1.test"},
    }

    with pytest.raises(RuntimeError, match="no native-extension identity"):
        build_campaign_manifest(
            run_kind="single_objective",
            budgets=[25],
            n_runs=1,
            problem_names=["problem"],
            optimizer_names=["optimizer"],
            optimizer_configurations=[],
            provenance=provenance,
        )


def test_installed_hola_with_unreadable_extension_fails_actionably(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"cannot be read.*maturin develop"):
        collect_provenance(
            tmp_path,
            dependency_versions={"hola-opt": "1.test"},
            native_extension_path=tmp_path / "hola_opt.abi3.so",
        )


def test_rust_build_definitions_are_hashed_but_target_manifests_are_not(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    member_manifest = repo_root / "hola-cli" / "Cargo.toml"
    target_manifest = repo_root / "target" / "generated" / "Cargo.toml"
    toolchain = repo_root / "rust-toolchain.toml"
    member_manifest.parent.mkdir(parents=True)
    target_manifest.parent.mkdir(parents=True)
    member_manifest.write_text("[package]\nname = 'member-one'\n")
    target_manifest.write_text("[package]\nname = 'generated-one'\n")
    toolchain.write_text("[toolchain]\nchannel = '1.87'\n")
    extension_path = tmp_path / "hola_opt.abi3.so"

    first = _collect_with_test_extension(repo_root, extension_path, b"native")
    member_manifest.write_text("[package]\nname = 'member-two'\n")
    second = _collect_with_test_extension(repo_root, extension_path, b"native")
    assert first["code"]["source_hash"] != second["code"]["source_hash"]
    assert first["lock_hash"] == second["lock_hash"]

    target_manifest.write_text("[package]\nname = 'generated-two'\n")
    third = _collect_with_test_extension(repo_root, extension_path, b"native")
    assert second["code"]["source_hash"] == third["code"]["source_hash"]

    toolchain.write_text("[toolchain]\nchannel = '1.88'\n")
    fourth = _collect_with_test_extension(repo_root, extension_path, b"native")
    assert third["code"]["source_hash"] != fourth["code"]["source_hash"]
    assert third["lock_hash"] != fourth["lock_hash"]


def test_resume_requires_an_exact_manifest_match(tmp_path) -> None:
    store = ResultStore(tmp_path)
    expected = _manifest()
    store.prepare_campaign(expected, resume=False)

    assert json.loads((tmp_path / MANIFEST_FILENAME).read_text()) == expected
    store.prepare_campaign(expected, resume=True)

    with pytest.raises(RuntimeError, match="changed fields: budgets"):
        store.prepare_campaign(_manifest(budgets=[20]), resume=True)


def test_resume_rejects_changed_resolved_optimizer_configuration(tmp_path) -> None:
    store = ResultStore(tmp_path)
    expected = _manifest()
    store.prepare_campaign(expected, resume=False)
    changed = copy.deepcopy(expected)
    changed["optimizer_configurations"][0]["by_budget"][0]["configuration"]["adapter"] = "changed"
    changed = attach_fingerprint(changed)

    with pytest.raises(RuntimeError, match="changed fields: optimizer_configurations"):
        store.prepare_campaign(changed, resume=True)


def test_resume_rejects_a_tampered_manifest(tmp_path) -> None:
    store = ResultStore(tmp_path)
    store.prepare_campaign(_manifest(), resume=False)
    manifest = json.loads(store.manifest_path.read_text())
    manifest["budgets"] = [999]
    store.manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="fingerprint"):
        store.prepare_campaign(_manifest(), resume=True)


def test_resume_rejects_legacy_results_without_a_manifest(tmp_path) -> None:
    (tmp_path / "single_objective.csv").write_text("problem,optimizer\n")

    with pytest.raises(RuntimeError, match="no campaign manifest"):
        ResultStore(tmp_path).prepare_campaign(_manifest(), resume=True)


def test_non_resume_requires_a_fresh_output_directory(tmp_path) -> None:
    store = ResultStore(tmp_path)
    store.prepare_campaign(_manifest(), resume=False)

    with pytest.raises(RuntimeError, match="nonempty directory"):
        store.prepare_campaign(_manifest(), resume=False)


def test_result_rows_require_manifest_preparation(tmp_path) -> None:
    with pytest.raises(RuntimeError, match=r"prepare_campaign\(\)"):
        ResultStore(tmp_path).append_single({})


def test_result_store_serializes_audit_fields(tmp_path) -> None:
    single_store = ResultStore(tmp_path / "single")
    single_store.prepare_campaign(_manifest(), resume=False)
    single_store.append_single(
        {
            "problem": "quadratic",
            "optimizer": "optimizer",
            "budget": 10,
            "run_id": 0,
            "seed": 123,
            "status": "success",
            "error": "",
            "optimizer_config": {"population": np.int64(10)},
            "n_evaluations": 10,
            "best_value": 0.0,
            "best_params": {"x": np.float64(0.25)},
            "wall_time_seconds": 0.1,
            "convergence_trace": np.array([1.0, 0.0]),
        }
    )
    multi_store = ResultStore(tmp_path / "multi")
    multi_store.prepare_campaign(_manifest(run_kind="multi_objective"), resume=False)
    multi_store.append_multi(
        {
            "problem": "tradeoff",
            "optimizer": "optimizer",
            "budget": 10,
            "run_id": 0,
            "seed": 123,
            "status": "success",
            "error": "",
            "optimizer_config": {"population": 10},
            "n_evaluations": 10,
            "pareto_front": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "decision_vectors": np.array([[0.0], [1.0]]),
            "normalized_hypervolume_gap": 0.0,
            "normalized_igd": 0.0,
            "spacing": 0.0,
            "wall_time_seconds": 0.1,
            "n_pareto_points": 2,
        }
    )

    single = single_store.load_single().iloc[0]
    multi = multi_store.load_multi().iloc[0]
    assert json.loads(single["optimizer_config"]) == {"population": 10}
    assert json.loads(single["best_params"]) == {"x": 0.25}
    assert json.loads(single["convergence_trace"]) == [1.0, 0.0]
    assert json.loads(multi["pareto_front"]) == [[0.0, 1.0], [1.0, 0.0]]
    assert json.loads(multi["decision_vectors"]) == [[0.0], [1.0]]
    assert single_store.completed_so_runs() == {("quadratic", "optimizer", 10, 0)}
    assert multi_store.completed_mo_runs() == {("tradeoff", "optimizer", 10, 0)}
