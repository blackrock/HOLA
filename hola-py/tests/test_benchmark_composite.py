# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Method-level provenance and virtual-union reporting checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from benchmarks.data.composite import (  # noqa: E402
    COMPOSITE_MANIFEST_FILENAME,
    SOURCE_MANIFEST_COLUMN,
    SOURCE_PATH_COLUMN,
    SOURCE_RESULTS_COLUMN,
    build_composite_manifest,
    load_reporting_results,
    validate_composite_manifest,
    write_composite_manifest,
)
from benchmarks.data.manifest import attach_fingerprint, build_campaign_manifest  # noqa: E402
from benchmarks.data.persistence import ResultStore  # noqa: E402
from benchmarks.data.seeding import make_hpo_split_seed, make_seed  # noqa: E402
from benchmarks.plotting.multi_objective import (  # noqa: E402
    representative_terminal_front_tables,
)

pytestmark = pytest.mark.benchmarks


def _provenance(label: str) -> dict[str, Any]:
    return {
        "code": {"commit": label, "dirty": False, "source_hash": f"source-{label}"},
        "lock_hash": "lock",
        "python": {"implementation": "CPython", "version": "test"},
        "platform": {"platform": "test", "machine": "test", "system": "test"},
        "dependencies": {"hola-opt": "test"},
        "native_extension": {
            "module": "hola_opt.hola_opt",
            "filename": "hola_opt.abi3.so",
            "byte_size": 4,
            "sha256": "0" * 64,
        },
    }


def _single_manifest(
    optimizers: list[str],
    *,
    budget: int = 2,
    campaign_configuration: dict[str, Any] | None = None,
    provenance_label: str = "source",
) -> dict[str, Any]:
    return build_campaign_manifest(
        run_kind="single_objective",
        budgets=[budget],
        n_runs=1,
        problem_names=["forrester_1d"],
        optimizer_names=optimizers,
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [
                    {
                        "budget": budget,
                        "configuration": {"adapter": optimizer, "budget": budget},
                    }
                ],
            }
            for optimizer in optimizers
        ],
        campaign_configuration=campaign_configuration,
        provenance=_provenance(provenance_label),
    )


def _single_row(optimizer: str, value: float, *, budget: int = 2) -> dict[str, Any]:
    return {
        "problem": "forrester_1d",
        "optimizer": optimizer,
        "budget": budget,
        "run_id": 0,
        "seed": make_seed("forrester_1d", budget, 0),
        "status": "success",
        "error": "",
        "optimizer_config": {"adapter": optimizer, "budget": budget},
        "n_evaluations": budget,
        "best_value": value,
        "best_params": {"x": 0.5},
        "wall_time_seconds": 0.1,
        "convergence_trace": [value] * budget,
    }


def _write_single_campaign(
    path: Path,
    values: dict[str, float],
    *,
    budget: int = 2,
    campaign_configuration: dict[str, Any] | None = None,
    provenance_label: str = "source",
) -> ResultStore:
    store = ResultStore(path)
    store.prepare_campaign(
        _single_manifest(
            list(values),
            budget=budget,
            campaign_configuration=campaign_configuration,
            provenance_label=provenance_label,
        ),
        resume=False,
    )
    for optimizer, value in values.items():
        store.append_single(_single_row(optimizer, value, budget=budget))
    return store


def _write_hpo_campaign(
    path: Path,
    values: dict[str, float],
    *,
    provenance_label: str,
) -> ResultStore:
    budget = 2
    problem = "gbr_diabetes_hpo"
    configuration = {"benchmark": "fixed-split"}
    manifest = build_campaign_manifest(
        run_kind="hpo",
        budgets=[budget],
        n_runs=1,
        problem_names=[problem],
        optimizer_names=list(values),
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [
                    {
                        "budget": budget,
                        "configuration": {"adapter": optimizer, "budget": budget},
                    }
                ],
            }
            for optimizer in values
        ],
        campaign_configuration=configuration,
        provenance=_provenance(provenance_label),
    )
    store = ResultStore(path)
    store.prepare_campaign(manifest, resume=False)
    for optimizer, value in values.items():
        store.append_hpo(
            {
                "problem": problem,
                "optimizer": optimizer,
                "budget": budget,
                "run_id": 0,
                "search_seed": make_seed(problem, budget, 0),
                "split_seed": make_hpo_split_seed(problem, 0),
                "status": "success",
                "error": "",
                "optimizer_config": {"adapter": optimizer, "budget": budget},
                "n_validation_evaluations": budget,
                "best_validation_r2": value,
                "best_params": {"depth": 2},
                "validation_trace": [value] * budget,
                "heldout_test_r2": value - 0.1,
                "n_heldout_evaluations": 1,
                "train_size": 10,
                "validation_size": 5,
                "test_size": 5,
                "wall_time_seconds": 0.1,
            }
        )
    return store


def _write_multi_campaign(
    path: Path,
    values: dict[str, float],
    *,
    provenance_label: str,
) -> ResultStore:
    budget = 2
    problem = "zdt3_30d"
    manifest = build_campaign_manifest(
        run_kind="multi_objective",
        budgets=[budget],
        n_runs=1,
        problem_names=[problem],
        optimizer_names=list(values),
        optimizer_configurations=[
            {
                "optimizer": optimizer,
                "by_budget": [
                    {
                        "budget": budget,
                        "configuration": {"adapter": optimizer, "budget": budget},
                    }
                ],
            }
            for optimizer in values
        ],
        provenance=_provenance(provenance_label),
    )
    store = ResultStore(path)
    store.prepare_campaign(manifest, resume=False)
    for optimizer, value in values.items():
        store.append_multi(
            {
                "problem": problem,
                "optimizer": optimizer,
                "budget": budget,
                "run_id": 0,
                "seed": make_seed(problem, budget, 0),
                "status": "success",
                "error": "",
                "optimizer_config": {"adapter": optimizer, "budget": budget},
                "n_evaluations": budget,
                "pareto_front": [[value, 1.0 - value]],
                "decision_vectors": [[0.5] * 30],
                "normalized_hypervolume_gap": value,
                "normalized_igd": value + 0.1,
                "spacing": 0.05,
                "wall_time_seconds": 0.1,
                "n_pareto_points": 1,
            }
        )
    return store


def _compose_single(tmp_path: Path) -> tuple[Path, ResultStore, ResultStore]:
    base = _write_single_campaign(
        tmp_path / "base",
        {"baseline": 1.0, "gmm": 100.0},
        provenance_label="base",
    )
    replacement = _write_single_campaign(
        tmp_path / "replacement",
        {"gmm": -1.0},
        provenance_label="replacement",
    )
    composite = tmp_path / "composite"
    write_composite_manifest(
        composite,
        base.output_dir,
        [(replacement.output_dir, ["gmm"])],
    )
    return composite, base, replacement


def test_composite_replaces_only_selected_method_and_records_row_sources(tmp_path: Path) -> None:
    composite, base, replacement = _compose_single(tmp_path)

    results = load_reporting_results(composite, "single_objective").set_index("optimizer")

    assert results.loc["baseline", "best_value"] == 1.0
    assert results.loc["gmm", "best_value"] == -1.0
    assert results.loc["baseline", SOURCE_PATH_COLUMN] == "../base"
    assert results.loc["gmm", SOURCE_PATH_COLUMN] == "../replacement"
    assert results.loc["baseline", SOURCE_MANIFEST_COLUMN] == base.load_manifest()["fingerprint"]
    assert results.loc["gmm", SOURCE_MANIFEST_COLUMN] == replacement.load_manifest()["fingerprint"]
    assert results[SOURCE_RESULTS_COLUMN].map(lambda value: len(value) == 64).all()
    assert not (composite / "single_objective.csv").exists()

    manifest = json.loads((composite / COMPOSITE_MANIFEST_FILENAME).read_text())
    assert manifest["optimizers"] == ["baseline", "gmm"]
    assert [source["optimizers"] for source in manifest["sources"]] == [
        ["baseline"],
        ["gmm"],
    ]
    assert all(not Path(source["path"]).is_absolute() for source in manifest["sources"])


def test_composite_paths_remain_valid_when_the_result_bundle_moves(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    composite, _, _ = _compose_single(bundle)
    moved = tmp_path / "moved"
    bundle.rename(moved)

    results = load_reporting_results(moved / composite.name, "single_objective")

    assert set(results["optimizer"]) == {"baseline", "gmm"}


def test_composite_refuses_manifest_source_and_result_tampering(tmp_path: Path) -> None:
    composite, base, replacement = _compose_single(tmp_path)
    composite_path = composite / COMPOSITE_MANIFEST_FILENAME

    payload = json.loads(composite_path.read_text())
    payload["optimizers"].reverse()
    composite_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="composite manifest fingerprint"):
        load_reporting_results(composite, "single_objective")

    payload["optimizers"].reverse()
    composite_path.write_text(json.dumps(attach_fingerprint(payload)))
    source_manifest = replacement.load_manifest()
    source_manifest["provenance"]["code"]["commit"] = "changed"
    replacement.manifest_path.write_text(json.dumps(attach_fingerprint(source_manifest)))
    with pytest.raises(RuntimeError, match="source campaign fingerprint changed"):
        load_reporting_results(composite, "single_objective")

    # Restore the exact source identity recorded by rebuilding the composition.
    replacement.manifest_path.write_text(
        json.dumps(_single_manifest(["gmm"], provenance_label="replacement"))
    )
    composite_path.unlink()
    write_composite_manifest(composite, base.output_dir, [(replacement.output_dir, ["gmm"])])
    with replacement.so_path.open("a") as result_file:
        result_file.write("\n")
    with pytest.raises(RuntimeError, match="source result artifact changed"):
        load_reporting_results(composite, "single_objective")


@pytest.mark.parametrize(
    ("replacement_budget", "replacement_configuration", "changed_field"),
    [
        (3, {"variant": "base"}, "budgets"),
        (2, {"variant": "other"}, "campaign_configuration"),
    ],
)
def test_composite_builder_rejects_incompatible_campaign_contracts(
    tmp_path: Path,
    replacement_budget: int,
    replacement_configuration: dict[str, Any],
    changed_field: str,
) -> None:
    base = _write_single_campaign(
        tmp_path / "base",
        {"baseline": 1.0, "gmm": 2.0},
        campaign_configuration={"variant": "base"},
    )
    replacement = _write_single_campaign(
        tmp_path / "replacement",
        {"gmm": -1.0},
        budget=replacement_budget,
        campaign_configuration=replacement_configuration,
    )

    with pytest.raises(RuntimeError, match=rf"changed fields:.*{changed_field}"):
        build_composite_manifest(
            tmp_path / "composite",
            base.output_dir,
            [(replacement.output_dir, ["gmm"])],
        )


def test_composite_builder_rejects_missing_and_duplicate_replacements(tmp_path: Path) -> None:
    base = _write_single_campaign(tmp_path / "base", {"baseline": 1.0, "gmm": 2.0})
    replacement = _write_single_campaign(tmp_path / "replacement", {"baseline": -1.0})

    with pytest.raises(RuntimeError, match="absent from the source campaign"):
        build_composite_manifest(
            tmp_path / "missing",
            base.output_dir,
            [(replacement.output_dir, ["gmm"])],
        )
    with pytest.raises(RuntimeError, match="assigned more than once"):
        build_composite_manifest(
            tmp_path / "duplicate",
            base.output_dir,
            [
                (base.output_dir, ["gmm"]),
                (base.output_dir, ["gmm"]),
            ],
        )


def test_composite_builder_rejects_incomplete_sources_before_writing(tmp_path: Path) -> None:
    base = _write_single_campaign(tmp_path / "base", {"baseline": 1.0, "gmm": 2.0})
    incomplete = ResultStore(tmp_path / "incomplete")
    incomplete.prepare_campaign(_single_manifest(["gmm"]), resume=False)
    output = tmp_path / "composite"

    with pytest.raises(RuntimeError, match="campaign results are incomplete"):
        write_composite_manifest(
            output,
            base.output_dir,
            [(incomplete.output_dir, ["gmm"])],
        )

    assert not output.exists()


def test_composite_validation_rejects_absolute_paths_and_ambiguous_directories(
    tmp_path: Path,
) -> None:
    composite, base, _ = _compose_single(tmp_path)
    composite_path = composite / COMPOSITE_MANIFEST_FILENAME
    payload = json.loads(composite_path.read_text())
    payload["sources"][0]["path"] = str(base.output_dir.resolve())
    payload = attach_fingerprint(payload)

    with pytest.raises(RuntimeError, match="relative POSIX"):
        validate_composite_manifest(payload)

    (composite / "campaign_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="ambiguous"):
        load_reporting_results(composite, "single_objective")


def test_composite_projection_preserves_source_row_contract_validation(tmp_path: Path) -> None:
    composite, _, replacement = _compose_single(tmp_path)
    rows = pd.read_csv(replacement.so_path)
    rows.loc[0, "seed"] = int(rows.loc[0, "seed"]) + 1
    rows.to_csv(replacement.so_path, index=False)
    payload = json.loads((composite / COMPOSITE_MANIFEST_FILENAME).read_text())
    payload["sources"][1]["result_file"]["byte_size"] = replacement.so_path.stat().st_size
    payload["sources"][1]["result_file"]["sha256"] = hashlib.sha256(
        replacement.so_path.read_bytes()
    ).hexdigest()
    (composite / COMPOSITE_MANIFEST_FILENAME).write_text(json.dumps(attach_fingerprint(payload)))

    with pytest.raises(RuntimeError, match="seed does not match deterministic derivation"):
        load_reporting_results(composite, "single_objective")


def test_hpo_composite_uses_the_same_dispatch_and_campaign_configuration(tmp_path: Path) -> None:
    base = _write_hpo_campaign(
        tmp_path / "base",
        {"baseline": 0.4, "gmm": 0.1},
        provenance_label="base",
    )
    replacement = _write_hpo_campaign(
        tmp_path / "replacement",
        {"gmm": 0.8},
        provenance_label="replacement",
    )
    composite = tmp_path / "composite"
    write_composite_manifest(
        composite,
        base.output_dir,
        [(replacement.output_dir, ["gmm"])],
    )

    results = load_reporting_results(composite, "hpo").set_index("optimizer")

    assert results.loc["baseline", "best_validation_r2"] == 0.4
    assert results.loc["gmm", "best_validation_r2"] == 0.8
    with pytest.raises(RuntimeError, match="expected a multi_objective report"):
        load_reporting_results(composite, "multi_objective")


def test_multi_composite_preserves_source_row_identity_in_front_audit(tmp_path: Path) -> None:
    base = _write_multi_campaign(
        tmp_path / "base",
        {"baseline": 0.4, "gmm": 0.8},
        provenance_label="base",
    )
    replacement = _write_multi_campaign(
        tmp_path / "replacement",
        {"gmm": 0.2},
        provenance_label="replacement",
    )
    composite = tmp_path / "composite"
    write_composite_manifest(
        composite,
        base.output_dir,
        [(replacement.output_dir, ["gmm"])],
    )

    results = load_reporting_results(composite, "multi_objective")
    selections, points = representative_terminal_front_tables(results, ["zdt3_30d"])
    gmm = selections.set_index("optimizer").loc["gmm"]

    assert gmm["normalized_hypervolume_gap"] == 0.2
    assert gmm["source_result_row_index"] == 0
    assert gmm[SOURCE_PATH_COLUMN] == "../replacement"
    assert gmm[SOURCE_MANIFEST_COLUMN] == replacement.load_manifest()["fingerprint"]
    assert len(gmm[SOURCE_RESULTS_COLUMN]) == 64
    assert len(points[points["optimizer"].eq("gmm")]) == 1
