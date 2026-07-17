# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Authenticated report composition across compatible benchmark campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.data.manifest import (
    MANIFEST_FILENAME,
    attach_fingerprint,
    manifest_fingerprint,
)
from benchmarks.data.persistence import ResultStore

COMPOSITE_MANIFEST_FILENAME = "composite_manifest.json"
COMPOSITE_VERSION = "1"

SOURCE_PATH_COLUMN = "source_campaign_path"
SOURCE_MANIFEST_COLUMN = "source_campaign_fingerprint"
SOURCE_RESULTS_COLUMN = "source_results_sha256"

_RESULT_FILENAMES = {
    "single_objective": "single_objective.csv",
    "multi_objective": "multi_objective.csv",
    "hpo": "hpo.csv",
}
_CONTRACT_FIELDS = {
    "protocol_version",
    "run_kind",
    "problems",
    "budgets",
    "n_runs",
    "campaign_configuration",
}
_SOURCE_FIELDS = {"path", "campaign_fingerprint", "result_file", "optimizers"}
_RESULT_FILE_FIELDS = {"filename", "byte_size", "sha256"}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _string_axis(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise RuntimeError(f"{label} must be a nonempty list of unique strings")
    return value


def _integer_axis(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
        or len(value) != len(set(value))
    ):
        raise RuntimeError(f"{label} must be a nonempty list of unique positive integers")
    return value


def _campaign_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    protocol_version = manifest.get("protocol_version")
    run_kind = manifest.get("run_kind")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise RuntimeError("campaign protocol_version must be a nonempty string")
    if run_kind not in _RESULT_FILENAMES:
        raise RuntimeError(f"campaign has unsupported run kind {run_kind!r}")
    problems = _string_axis(manifest.get("problems"), "campaign problems")
    budgets = _integer_axis(manifest.get("budgets"), "campaign budgets")
    n_runs = manifest.get("n_runs")
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs <= 0:
        raise RuntimeError("campaign n_runs must be a positive integer")
    campaign_configuration = manifest.get("campaign_configuration")
    if campaign_configuration is not None and not isinstance(campaign_configuration, dict):
        raise RuntimeError("campaign_configuration must be an object when present")
    return {
        "protocol_version": protocol_version,
        "run_kind": run_kind,
        "problems": list(problems),
        "budgets": list(budgets),
        "n_runs": n_runs,
        "campaign_configuration": campaign_configuration,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _source_store(source_dir: Path) -> ResultStore:
    if not source_dir.is_dir():
        raise RuntimeError(f"composite source campaign directory is missing: {source_dir}")
    if (source_dir / COMPOSITE_MANIFEST_FILENAME).exists():
        raise RuntimeError("a composite source must be an ordinary benchmark campaign")
    return ResultStore(source_dir)


def _artifact_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with path.open("rb") as result_file:
            while chunk := result_file.read(1024 * 1024):
                digest.update(chunk)
                byte_size += len(chunk)
    except OSError as error:
        raise RuntimeError(f"cannot read benchmark result artifact {path}: {error}") from error
    return {
        "filename": path.name,
        "byte_size": byte_size,
        "sha256": digest.hexdigest(),
    }


def _source_entry(
    composite_dir: Path,
    source_dir: Path,
    source_manifest: dict[str, Any],
    optimizers: list[str],
    run_kind: str,
) -> dict[str, Any]:
    result_path = source_dir / _RESULT_FILENAMES[run_kind]
    if not result_path.is_file():
        raise RuntimeError(f"source campaign result artifact is missing: {result_path}")
    relative_path = Path(os.path.relpath(source_dir, start=composite_dir)).as_posix()
    return {
        "path": relative_path,
        "campaign_fingerprint": source_manifest["fingerprint"],
        "result_file": _artifact_identity(result_path),
        "optimizers": optimizers,
    }


def validate_composite_manifest(manifest: dict[str, Any]) -> None:
    """Reject malformed, altered, ambiguous, or incomplete compositions."""
    expected_fingerprint = manifest_fingerprint(manifest)
    if manifest.get("fingerprint") != expected_fingerprint:
        raise RuntimeError("composite manifest fingerprint is missing or invalid")
    expected_fields = {
        "composite_version",
        "contract",
        "optimizers",
        "sources",
        "fingerprint",
    }
    if set(manifest) != expected_fields:
        raise RuntimeError("composite manifest has unexpected or missing fields")
    if manifest["composite_version"] != COMPOSITE_VERSION:
        raise RuntimeError(
            f"unsupported composite manifest version {manifest['composite_version']!r}"
        )

    contract = manifest["contract"]
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_FIELDS:
        raise RuntimeError("composite manifest has a malformed campaign contract")
    if not isinstance(contract["protocol_version"], str) or not contract["protocol_version"]:
        raise RuntimeError("composite protocol_version must be a nonempty string")
    run_kind = contract["run_kind"]
    if run_kind not in _RESULT_FILENAMES:
        raise RuntimeError(f"composite manifest has unsupported run kind {run_kind!r}")
    _string_axis(contract["problems"], "composite problems")
    _integer_axis(contract["budgets"], "composite budgets")
    n_runs = contract["n_runs"]
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs <= 0:
        raise RuntimeError("composite n_runs must be a positive integer")
    if contract["campaign_configuration"] is not None and not isinstance(
        contract["campaign_configuration"], dict
    ):
        raise RuntimeError("composite campaign_configuration must be an object or null")

    target_optimizers = _string_axis(manifest["optimizers"], "composite optimizers")
    sources = manifest["sources"]
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("composite sources must be a nonempty list")

    assigned: set[str] = set()
    source_paths: set[str] = set()
    expected_filename = _RESULT_FILENAMES[run_kind]
    for source in sources:
        if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
            raise RuntimeError("composite manifest has a malformed source")
        source_path = source["path"]
        if (
            not isinstance(source_path, str)
            or not source_path
            or source_path == "."
            or Path(source_path).is_absolute()
            or "\\" in source_path
        ):
            raise RuntimeError("composite source paths must be relative POSIX directory paths")
        if source_path in source_paths:
            raise RuntimeError(f"composite source path is repeated: {source_path}")
        source_paths.add(source_path)
        if not _is_sha256(source["campaign_fingerprint"]):
            raise RuntimeError("composite source has an invalid campaign fingerprint")

        artifact = source["result_file"]
        if not isinstance(artifact, dict) or set(artifact) != _RESULT_FILE_FIELDS:
            raise RuntimeError("composite source has a malformed result-file identity")
        if artifact["filename"] != expected_filename:
            raise RuntimeError(f"composite source result filename must be {expected_filename!r}")
        byte_size = artifact["byte_size"]
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
            raise RuntimeError("composite source result byte_size must be nonnegative")
        if not _is_sha256(artifact["sha256"]):
            raise RuntimeError("composite source has an invalid result SHA-256")

        selected = _string_axis(source["optimizers"], "composite source optimizers")
        unknown = [optimizer for optimizer in selected if optimizer not in target_optimizers]
        if unknown:
            raise RuntimeError(
                "composite source assigns optimizer(s) outside the target campaign: "
                + ", ".join(unknown)
            )
        overlap = assigned.intersection(selected)
        if overlap:
            raise RuntimeError(
                "composite optimizer assignments overlap: " + ", ".join(sorted(overlap))
            )
        assigned.update(selected)

    missing = [optimizer for optimizer in target_optimizers if optimizer not in assigned]
    if missing:
        raise RuntimeError("composite optimizer assignments are incomplete: " + ", ".join(missing))


def build_composite_manifest(
    composite_dir: Path,
    base_results_dir: Path,
    replacements: Sequence[tuple[Path, Sequence[str]]],
) -> dict[str, Any]:
    """Describe a base campaign with method-level source replacements."""
    composite_dir = composite_dir.resolve()
    base_results_dir = base_results_dir.resolve()
    base_store = _source_store(base_results_dir)
    base_manifest = base_store.load_manifest()
    contract = _campaign_contract(base_manifest)
    target_optimizers = _string_axis(base_manifest.get("optimizers"), "base optimizers")

    assignments = {optimizer: base_results_dir for optimizer in target_optimizers}
    explicitly_replaced: set[str] = set()
    for replacement_dir, replacement_optimizers in replacements:
        selected = list(replacement_optimizers)
        _string_axis(selected, "replacement optimizers")
        unknown = [optimizer for optimizer in selected if optimizer not in target_optimizers]
        if unknown:
            raise RuntimeError(
                "replacement optimizer(s) are absent from the base campaign: " + ", ".join(unknown)
            )
        overlap = explicitly_replaced.intersection(selected)
        if overlap:
            raise RuntimeError(
                "optimizer replacement is assigned more than once: " + ", ".join(sorted(overlap))
            )
        source_dir = replacement_dir.resolve()
        for optimizer in selected:
            assignments[optimizer] = source_dir
        explicitly_replaced.update(selected)

    grouped: dict[Path, list[str]] = {}
    for optimizer in target_optimizers:
        grouped.setdefault(assignments[optimizer], []).append(optimizer)

    sources = []
    for source_dir, selected in grouped.items():
        store = _source_store(source_dir)
        source_manifest = store.load_manifest(expected_run_kind=contract["run_kind"])
        source_contract = _campaign_contract(source_manifest)
        if source_contract != contract:
            changed = [
                field
                for field in sorted(_CONTRACT_FIELDS)
                if source_contract[field] != contract[field]
            ]
            raise RuntimeError(
                "source campaign is incompatible with the composite contract "
                f"(changed fields: {', '.join(changed)})"
            )
        source_optimizers = _string_axis(
            source_manifest.get("optimizers"), "source campaign optimizers"
        )
        missing = [optimizer for optimizer in selected if optimizer not in source_optimizers]
        if missing:
            raise RuntimeError(
                "assigned optimizer(s) are absent from the source campaign: " + ", ".join(missing)
            )
        # A composite is a reporting artifact, so unlike an in-progress
        # campaign it must be usable at creation time. Validate the selected
        # projection before pinning the result-file digest.
        store.load_complete_selected(contract["run_kind"], selected)
        sources.append(
            _source_entry(
                composite_dir,
                source_dir,
                source_manifest,
                selected,
                contract["run_kind"],
            )
        )

    manifest = attach_fingerprint(
        {
            "composite_version": COMPOSITE_VERSION,
            "contract": contract,
            "optimizers": list(target_optimizers),
            "sources": sources,
        }
    )
    validate_composite_manifest(manifest)
    return manifest


def write_composite_manifest(
    output_dir: Path,
    base_results_dir: Path,
    replacements: Sequence[tuple[Path, Sequence[str]]],
) -> dict[str, Any]:
    """Atomically write a composite manifest into a fresh report directory."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"composite output directory must be empty: {output_dir}")
    manifest = build_composite_manifest(output_dir, base_results_dir, replacements)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"composite output directory must be empty: {output_dir}")
    manifest_path = output_dir / COMPOSITE_MANIFEST_FILENAME
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_dir,
        prefix=f".{COMPOSITE_MANIFEST_FILENAME}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    try:
        os.link(temporary, manifest_path)
    except FileExistsError as error:
        raise RuntimeError(f"composite manifest already exists at {manifest_path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _load_ordinary_results(results_dir: Path, run_kind: str) -> pd.DataFrame:
    store = ResultStore(results_dir)
    if run_kind == "single_objective":
        return store.load_complete_single()
    if run_kind == "multi_objective":
        return store.load_complete_multi()
    if run_kind == "hpo":
        return store.load_complete_hpo()
    raise RuntimeError(f"unsupported benchmark run kind {run_kind!r}")


def _verify_source(
    composite_dir: Path,
    source: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source_dir = (composite_dir / source["path"]).resolve()
    store = _source_store(source_dir)
    source_manifest = store.load_manifest(expected_run_kind=contract["run_kind"])
    if source_manifest["fingerprint"] != source["campaign_fingerprint"]:
        raise RuntimeError(
            f"source campaign fingerprint changed for composite source {source['path']!r}"
        )
    if _campaign_contract(source_manifest) != contract:
        raise RuntimeError(
            f"source campaign contract changed for composite source {source['path']!r}"
        )
    source_optimizers = _string_axis(
        source_manifest.get("optimizers"), "source campaign optimizers"
    )
    missing = [
        optimizer for optimizer in source["optimizers"] if optimizer not in source_optimizers
    ]
    if missing:
        raise RuntimeError(
            f"composite source {source['path']!r} no longer declares optimizer(s): "
            + ", ".join(missing)
        )

    result_path = source_dir / source["result_file"]["filename"]
    observed_artifact = _artifact_identity(result_path)
    if observed_artifact != source["result_file"]:
        raise RuntimeError(
            f"source result artifact changed for composite source {source['path']!r}"
        )
    return source_dir, source_manifest


def _validate_composed_rows(results: pd.DataFrame, manifest: dict[str, Any]) -> None:
    contract = manifest["contract"]
    optimizers = manifest["optimizers"]
    key_columns = ["problem", "optimizer", "budget", "run_id"]
    duplicate_mask = results.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        raise RuntimeError("composite results contain overlapping run keys")
    expected_keys = set(
        product(
            contract["problems"],
            optimizers,
            contract["budgets"],
            range(contract["n_runs"]),
        )
    )
    observed_keys = set(results[key_columns].itertuples(index=False, name=None))
    if observed_keys != expected_keys:
        missing = expected_keys - observed_keys
        extra = observed_keys - expected_keys
        raise RuntimeError(
            "composite results do not cover the target Cartesian product "
            f"(missing {len(missing)}, unexpected {len(extra)})"
        )

    seed_columns = ("search_seed", "split_seed") if contract["run_kind"] == "hpo" else ("seed",)
    pairing_keys = ["problem", "budget", "run_id"]
    for column in seed_columns:
        paired_counts = results.groupby(pairing_keys, dropna=False)[column].nunique(dropna=False)
        if (paired_counts != 1).any():
            raise RuntimeError(f"composite results violate cross-optimizer {column} pairing")


def load_composite_results(composite_dir: Path, expected_run_kind: str) -> pd.DataFrame:
    """Load a validated virtual union without copying source result files."""
    manifest_path = composite_dir / COMPOSITE_MANIFEST_FILENAME
    manifest = _read_json_object(manifest_path, "composite manifest")
    validate_composite_manifest(manifest)
    contract = manifest["contract"]
    if contract["run_kind"] != expected_run_kind:
        raise RuntimeError(f"expected a {expected_run_kind} report, found {contract['run_kind']!r}")

    frames: list[pd.DataFrame] = []
    for source in manifest["sources"]:
        source_dir, source_manifest = _verify_source(composite_dir, source, contract)
        frame = ResultStore(source_dir).load_complete_selected(
            expected_run_kind,
            source["optimizers"],
        )
        frame[SOURCE_PATH_COLUMN] = source["path"]
        frame[SOURCE_MANIFEST_COLUMN] = source_manifest["fingerprint"]
        frame[SOURCE_RESULTS_COLUMN] = source["result_file"]["sha256"]
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    _validate_composed_rows(results, manifest)
    return results


def load_reporting_results(results_dir: Path, expected_run_kind: str) -> pd.DataFrame:
    """Load exactly one ordinary campaign or one composite report source."""
    if expected_run_kind not in _RESULT_FILENAMES:
        raise RuntimeError(f"unsupported benchmark run kind {expected_run_kind!r}")
    if not results_dir.is_dir():
        raise RuntimeError(f"benchmark results directory is missing: {results_dir}")
    campaign_exists = (results_dir / MANIFEST_FILENAME).exists()
    composite_exists = (results_dir / COMPOSITE_MANIFEST_FILENAME).exists()
    if campaign_exists and composite_exists:
        raise RuntimeError(
            "benchmark results directory is ambiguous: it contains campaign and composite manifests"
        )
    if not campaign_exists and not composite_exists:
        raise RuntimeError(
            "benchmark results directory contains neither a campaign nor composite manifest"
        )
    if campaign_exists:
        return _load_ordinary_results(results_dir, expected_run_kind)
    return load_composite_results(results_dir, expected_run_kind)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose compatible benchmark campaigns for authenticated reporting"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-results-dir", type=Path, required=True)
    parser.add_argument(
        "--replacement",
        action="append",
        nargs=2,
        default=[],
        metavar=("RESULTS_DIR", "OPTIMIZERS"),
        help="Replacement campaign and comma-separated optimizer names; may be repeated",
    )
    args = parser.parse_args()
    replacements = [
        (Path(source), [optimizer.strip() for optimizer in names.split(",")])
        for source, names in args.replacement
    ]
    manifest = write_composite_manifest(
        args.output_dir,
        args.base_results_dir,
        replacements,
    )
    print(f"Wrote {args.output_dir / COMPOSITE_MANIFEST_FILENAME} ({manifest['fingerprint']})")


if __name__ == "__main__":
    main()
