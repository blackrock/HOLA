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

"""CSV-based result persistence with resume support."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import threading
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

from benchmarks.data.manifest import MANIFEST_FILENAME, validate_manifest
from benchmarks.data.schema import HPO_COLUMNS, MO_COLUMNS, SO_COLUMNS
from benchmarks.data.seeding import make_hpo_split_seed, make_seed


class _ResultRow(Protocol):
    """Typed view of the union of persisted benchmark result schemas."""

    problem: Any
    optimizer: Any
    budget: Any
    run_id: Any
    status: Any
    optimizer_config: Any
    seed: Any
    n_evaluations: Any
    convergence_trace: Any
    search_seed: Any
    split_seed: Any
    n_validation_evaluations: Any
    n_heldout_evaluations: Any
    validation_trace: Any


class ResultStore:
    """Thread-safe CSV result store with resume support."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._campaign_prepared = False
        self._run_kind: str | None = None

    @property
    def so_path(self) -> Path:
        return self.output_dir / "single_objective.csv"

    @property
    def mo_path(self) -> Path:
        return self.output_dir / "multi_objective.csv"

    @property
    def hpo_path(self) -> Path:
        return self.output_dir / "hpo.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / MANIFEST_FILENAME

    def load_manifest(self, *, expected_run_kind: str | None = None) -> dict[str, Any]:
        """Read and authenticate the campaign manifest used for reporting."""
        if not self.manifest_path.exists():
            raise RuntimeError(f"campaign manifest is missing from {self.output_dir}")
        try:
            observed = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read campaign manifest: {error}") from error
        if not isinstance(observed, dict):
            raise RuntimeError("campaign manifest must contain a JSON object")
        validate_manifest(observed)
        run_kind = observed.get("run_kind")
        if expected_run_kind is not None and run_kind != expected_run_kind:
            raise RuntimeError(f"expected a {expected_run_kind} campaign, found {run_kind!r}")
        return observed

    def load_complete_single(self) -> pd.DataFrame:
        """Load a complete, authenticated single-objective campaign."""
        return self._load_complete_campaign("single_objective", self.load_single())

    def load_complete_multi(self) -> pd.DataFrame:
        """Load a complete, authenticated multi-objective campaign."""
        return self._load_complete_campaign("multi_objective", self.load_multi())

    def load_complete_hpo(self) -> pd.DataFrame:
        """Load a complete, authenticated practical-HPO campaign."""
        return self._load_complete_campaign("hpo", self.load_hpo())

    def _load_complete_campaign(
        self,
        expected_run_kind: str,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate exact manifest Cartesian coverage before analysis."""
        manifest = self.load_manifest(expected_run_kind=expected_run_kind)
        problems = self._manifest_string_axis(manifest, "problems")
        optimizers = self._manifest_string_axis(manifest, "optimizers")
        budgets = self._manifest_integer_axis(manifest, "budgets")
        n_runs = manifest.get("n_runs")
        if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs <= 0:
            raise RuntimeError("campaign manifest n_runs must be a positive integer")

        key_columns = ["problem", "optimizer", "budget", "run_id"]
        required_columns = {*key_columns, "status", "optimizer_config"}
        if expected_run_kind == "hpo":
            required_columns.update(
                {
                    "search_seed",
                    "split_seed",
                    "n_validation_evaluations",
                    "n_heldout_evaluations",
                    "validation_trace",
                }
            )
        else:
            required_columns.update({"seed", "n_evaluations"})
            if expected_run_kind == "single_objective":
                required_columns.add("convergence_trace")
        missing_columns = required_columns - set(results.columns)
        if missing_columns:
            raise RuntimeError(
                "campaign results are missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        if results[key_columns].isna().any(axis=None):
            raise RuntimeError("campaign results contain missing run-key values")

        canonical = results.copy()
        for column in ("budget", "run_id"):
            numeric = pd.to_numeric(canonical[column], errors="coerce")
            invalid = numeric.isna() | ~numeric.map(math.isfinite) | numeric.mod(1).ne(0)
            if invalid.any():
                raise RuntimeError(f"campaign results contain non-integer {column} values")
            canonical[column] = numeric.astype(int)
        seed_columns = ("search_seed", "split_seed") if expected_run_kind == "hpo" else ("seed",)
        for column in seed_columns:
            numeric = pd.to_numeric(canonical[column], errors="coerce")
            invalid = numeric.isna() | ~numeric.map(math.isfinite) | numeric.mod(1).ne(0)
            invalid |= (numeric < 0) | (numeric > 2**32 - 1)
            if invalid.any():
                raise RuntimeError(
                    f"campaign results contain invalid unsigned 32-bit {column} values"
                )
            canonical[column] = numeric.astype("uint64")
        for column in ("problem", "optimizer"):
            if not canonical[column].map(lambda value: isinstance(value, str)).all():
                raise RuntimeError(f"campaign results contain non-string {column} values")

        statuses = canonical["status"]
        invalid_status = statuses.isna() | ~statuses.isin({"success", "error"})
        if invalid_status.any():
            examples = sorted({repr(value) for value in statuses[invalid_status]})
            raise RuntimeError(
                "campaign results contain malformed statuses: " + ", ".join(examples[:3])
            )

        duplicate_mask = canonical.duplicated(key_columns, keep=False)
        if duplicate_mask.any():
            duplicate_keys = self._format_keys(canonical.loc[duplicate_mask, key_columns])
            raise RuntimeError(
                "campaign results contain duplicate run keys; examples: " + duplicate_keys
            )

        expected_keys = set(product(problems, optimizers, budgets, range(n_runs)))
        observed_keys = set(canonical[key_columns].itertuples(index=False, name=None))
        extra_keys = observed_keys - expected_keys
        if extra_keys:
            raise RuntimeError(
                f"campaign results contain {len(extra_keys)} unexpected run key(s); "
                f"examples: {self._format_key_tuples(extra_keys)}"
            )
        missing_keys = expected_keys - observed_keys
        if missing_keys:
            raise RuntimeError(
                f"campaign results are incomplete: missing {len(missing_keys)} expected "
                f"run(s); examples: {self._format_key_tuples(missing_keys)}"
            )
        configurations = self._manifest_optimizer_configurations(
            manifest,
            optimizers,
            budgets,
        )
        self._validate_row_configurations(canonical, configurations)
        if expected_run_kind == "hpo":
            self._validate_hpo_contract(canonical)
        else:
            self._validate_standard_seed_contract(canonical)
            if expected_run_kind == "single_objective":
                self._validate_single_objective_contract(canonical, configurations)
            else:
                self._validate_multi_objective_contract(canonical)
        return canonical

    @staticmethod
    def _manifest_optimizer_configurations(
        manifest: dict[str, Any],
        optimizers: list[str],
        budgets: list[int],
    ) -> dict[tuple[str, int], dict[str, Any]]:
        """Return the manifest's complete optimizer-by-budget configuration map."""
        entries = manifest.get("optimizer_configurations")
        if not isinstance(entries, list):
            raise RuntimeError("campaign manifest optimizer_configurations must be a list")

        configurations: dict[tuple[str, int], dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError("campaign manifest has a malformed optimizer configuration")
            optimizer = entry.get("optimizer")
            by_budget = entry.get("by_budget")
            if not isinstance(optimizer, str) or not isinstance(by_budget, list):
                raise RuntimeError("campaign manifest has a malformed optimizer configuration")
            for item in by_budget:
                if not isinstance(item, dict):
                    raise RuntimeError("campaign manifest has a malformed budget configuration")
                budget = item.get("budget")
                configuration = item.get("configuration")
                if (
                    isinstance(budget, bool)
                    or not isinstance(budget, int)
                    or not isinstance(configuration, dict)
                ):
                    raise RuntimeError("campaign manifest has a malformed budget configuration")
                key = (optimizer, budget)
                if key in configurations:
                    raise RuntimeError(f"campaign manifest repeats optimizer configuration {key!r}")
                configurations[key] = configuration

        expected = set(product(optimizers, budgets))
        observed = set(configurations)
        if observed != expected:
            missing = expected - observed
            extra = observed - expected
            detail = []
            if missing:
                detail.append(f"missing {len(missing)}")
            if extra:
                detail.append(f"unexpected {len(extra)}")
            raise RuntimeError(
                "campaign manifest optimizer configurations do not cover the declared "
                f"Cartesian product ({', '.join(detail)})"
            )
        return configurations

    @staticmethod
    def _result_rows(results: pd.DataFrame) -> Iterator[_ResultRow]:
        for row in results.itertuples(index=False):
            yield cast(_ResultRow, row)

    @staticmethod
    def _row_key(row: _ResultRow) -> tuple[str, str, int, int]:
        return (
            str(row.problem),
            str(row.optimizer),
            int(row.budget),
            int(row.run_id),
        )

    @staticmethod
    def _parse_json_container(
        value: Any,
        expected_type: type[dict] | type[list],
        column: str,
        key: tuple[str, str, int, int],
        *,
        allow_missing: bool = False,
    ) -> dict[str, Any] | list[Any] | None:
        if value is None or (not isinstance(value, (str, dict, list)) and bool(pd.isna(value))):
            if allow_missing:
                return None
            raise RuntimeError(f"campaign result {key!r} is missing {column}")
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"campaign result {key!r} has malformed JSON in {column}"
                ) from error
        else:
            parsed = value
        if not isinstance(parsed, expected_type):
            expected_name = "object" if expected_type is dict else "array"
            raise RuntimeError(
                f"campaign result {key!r} {column} must contain a JSON {expected_name}"
            )
        return parsed

    @staticmethod
    def _integer_value(
        value: Any,
        column: str,
        key: tuple[str, str, int, int],
        *,
        allow_missing: bool = False,
    ) -> int | None:
        if value is None or (not isinstance(value, str) and bool(pd.isna(value))):
            if allow_missing:
                return None
            raise RuntimeError(f"campaign result {key!r} is missing {column}")
        if isinstance(value, bool):
            raise RuntimeError(f"campaign result {key!r} has non-integer {column}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"campaign result {key!r} has non-integer {column}") from error
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise RuntimeError(f"campaign result {key!r} has invalid {column}")
        return int(numeric)

    @classmethod
    def _validate_row_configurations(
        cls,
        results: pd.DataFrame,
        configurations: dict[tuple[str, int], dict[str, Any]],
    ) -> None:
        """Require every stored row configuration to equal its frozen manifest entry."""
        for row in cls._result_rows(results):
            key = cls._row_key(row)
            observed = cls._parse_json_container(
                row.optimizer_config,
                dict,
                "optimizer_config",
                key,
            )
            expected = configurations[(key[1], key[2])]
            if observed != expected:
                raise RuntimeError(
                    f"campaign result {key!r} optimizer_config does not match the manifest"
                )

    @classmethod
    def _validate_standard_seed_contract(cls, results: pd.DataFrame) -> None:
        """Validate deterministic SO/MO seeds, including error outcomes."""
        for row in cls._result_rows(results):
            key = cls._row_key(row)
            expected = make_seed(key[0], key[2], key[3])
            if int(row.seed) != expected:
                raise RuntimeError(
                    f"campaign result {key!r} seed does not match deterministic derivation"
                )

    @classmethod
    def _validate_single_objective_contract(
        cls,
        results: pd.DataFrame,
        configurations: dict[tuple[str, int], dict[str, Any]],
    ) -> None:
        """Validate successful SO evaluation and convergence-trace counts."""
        for row in cls._result_rows(results):
            key = cls._row_key(row)
            status = str(row.status)
            actual = cls._integer_value(
                row.n_evaluations,
                "n_evaluations",
                key,
                allow_missing=status == "error",
            )
            if status != "success":
                continue

            configuration = configurations[(key[1], key[2])]
            multiplier = configuration.get("evaluation_multiplier")
            if multiplier is None:
                if key[1] == "Random x2":
                    raise RuntimeError("Random x2 manifest configuration lacks its 2x multiplier")
                expected = key[2]
            else:
                if (
                    isinstance(multiplier, bool)
                    or not isinstance(multiplier, int)
                    or multiplier < 1
                ):
                    raise RuntimeError(
                        f"manifest configuration for {key[1]!r} has an invalid "
                        "evaluation multiplier"
                    )
                expected = multiplier * key[2]
                if configuration.get("actual_budget") != expected:
                    raise RuntimeError(
                        f"manifest configuration for {key[1]!r} has an inconsistent actual budget"
                    )
                if key[1] == "Random x2" and multiplier != 2:
                    raise RuntimeError(
                        "Random x2 manifest configuration must declare 2x evaluations"
                    )
            if actual != expected:
                raise RuntimeError(
                    f"campaign result {key!r} used {actual} evaluations; expected {expected}"
                )
            trace = cls._parse_json_container(
                row.convergence_trace,
                list,
                "convergence_trace",
                key,
            )
            assert isinstance(trace, list)
            if len(trace) != expected:
                raise RuntimeError(
                    f"campaign result {key!r} convergence_trace has {len(trace)} entries; "
                    f"expected {expected}"
                )

    @classmethod
    def _validate_multi_objective_contract(cls, results: pd.DataFrame) -> None:
        """Validate successful MO evaluation counts."""
        for row in cls._result_rows(results):
            key = cls._row_key(row)
            status = str(row.status)
            actual = cls._integer_value(
                row.n_evaluations,
                "n_evaluations",
                key,
                allow_missing=status == "error",
            )
            if status == "success" and actual != key[2]:
                raise RuntimeError(
                    f"campaign result {key!r} used {actual} evaluations; expected {key[2]}"
                )

    @classmethod
    def _validate_hpo_contract(cls, results: pd.DataFrame) -> None:
        """Validate HPO seed scopes, validation calls, traces, and sealed-test access."""
        for row in cls._result_rows(results):
            key = cls._row_key(row)
            status = str(row.status)
            expected_search_seed = make_seed(key[0], key[2], key[3])
            expected_split_seed = make_hpo_split_seed(key[0], key[3])
            if int(row.search_seed) != expected_search_seed:
                raise RuntimeError(
                    f"campaign result {key!r} search_seed does not match deterministic derivation"
                )
            if int(row.split_seed) != expected_split_seed:
                raise RuntimeError(
                    f"campaign result {key!r} split_seed does not match deterministic derivation"
                )

            validation_calls = cls._integer_value(
                row.n_validation_evaluations,
                "n_validation_evaluations",
                key,
                allow_missing=status == "error",
            )
            heldout_calls = cls._integer_value(
                row.n_heldout_evaluations,
                "n_heldout_evaluations",
                key,
            )
            if validation_calls is not None and validation_calls > key[2]:
                raise RuntimeError(
                    f"campaign result {key!r} exceeds its validation-evaluation budget"
                )
            if heldout_calls not in {0, 1}:
                raise RuntimeError(
                    f"campaign result {key!r} must use zero or one held-out evaluations"
                )
            if heldout_calls == 1 and validation_calls != key[2]:
                raise RuntimeError(
                    f"campaign result {key!r} accessed held-out data before completing validation"
                )

            trace = cls._parse_json_container(
                row.validation_trace,
                list,
                "validation_trace",
                key,
                allow_missing=status == "error",
            )
            if trace is not None and len(trace) != validation_calls:
                raise RuntimeError(
                    f"campaign result {key!r} validation_trace length does not match its calls"
                )
            if status == "success":
                if validation_calls != key[2]:
                    raise RuntimeError(
                        f"campaign result {key!r} used {validation_calls} validation evaluations; "
                        f"expected {key[2]}"
                    )
                if heldout_calls != 1:
                    raise RuntimeError(
                        f"campaign result {key!r} must use exactly one held-out evaluation"
                    )
                if trace is None or len(trace) != key[2]:
                    raise RuntimeError(
                        f"campaign result {key!r} validation_trace must contain {key[2]} entries"
                    )

    @staticmethod
    def _manifest_string_axis(manifest: dict[str, Any], field: str) -> list[str]:
        values = manifest.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise RuntimeError(
                f"campaign manifest {field} must be a nonempty list of unique strings"
            )
        return values

    @staticmethod
    def _manifest_integer_axis(manifest: dict[str, Any], field: str) -> list[int]:
        values = manifest.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            )
            or len(values) != len(set(values))
        ):
            raise RuntimeError(
                f"campaign manifest {field} must be a nonempty list of unique positive integers"
            )
        return values

    @staticmethod
    def _format_keys(keys: pd.DataFrame) -> str:
        return ResultStore._format_key_tuples(set(keys.itertuples(index=False, name=None)))

    @staticmethod
    def _format_key_tuples(keys: set[tuple[Any, ...]]) -> str:
        return ", ".join(repr(key) for key in sorted(keys, key=repr)[:3])

    def prepare_campaign(self, expected: dict[str, Any], *, resume: bool) -> None:
        """Create or validate the immutable manifest before reading/writing rows."""
        self._campaign_prepared = False
        self._run_kind = None
        validate_manifest(expected)
        entries = list(self.output_dir.iterdir())

        if not resume:
            if entries:
                raise RuntimeError(
                    f"refusing a non-resume run in nonempty directory {self.output_dir}; "
                    "choose a fresh output directory"
                )
            self._write_manifest(expected)
            self._campaign_prepared = True
            self._run_kind = str(expected["run_kind"])
            return

        if not self.manifest_path.exists():
            if entries:
                raise RuntimeError(
                    f"cannot resume {self.output_dir}: existing results have no campaign manifest"
                )
            self._write_manifest(expected)
            self._campaign_prepared = True
            self._run_kind = str(expected["run_kind"])
            return

        try:
            observed = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read campaign manifest: {error}") from error
        if not isinstance(observed, dict):
            raise RuntimeError("campaign manifest must contain a JSON object")
        validate_manifest(observed)
        if observed != expected:
            mismatches = sorted(
                key
                for key in set(observed) | set(expected)
                if key != "fingerprint" and observed.get(key) != expected.get(key)
            )
            raise RuntimeError(
                "campaign manifest mismatch; refusing unsafe resume "
                f"(changed fields: {', '.join(mismatches)})"
            )
        self._campaign_prepared = True
        self._run_kind = str(expected["run_kind"])

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomically create the manifest without replacing an existing one."""
        if self.manifest_path.exists():
            raise RuntimeError(f"campaign manifest already exists at {self.manifest_path}")
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.output_dir,
            prefix=f".{MANIFEST_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, self.manifest_path)
        except FileExistsError as error:
            raise RuntimeError(
                f"campaign manifest already exists at {self.manifest_path}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    def append_single(self, row: dict[str, Any]) -> None:
        """Append a single-objective result row."""
        self._require_campaign("single_objective")
        row = self._serialize_json_fields(
            row,
            ("optimizer_config", "best_params", "convergence_trace"),
        )
        self._append_row(self.so_path, SO_COLUMNS, row)

    def append_multi(self, row: dict[str, Any]) -> None:
        """Append a multi-objective result row."""
        self._require_campaign("multi_objective")
        row = self._serialize_json_fields(
            row,
            ("optimizer_config", "pareto_front", "decision_vectors"),
        )
        self._append_row(self.mo_path, MO_COLUMNS, row)

    def append_hpo(self, row: dict[str, Any]) -> None:
        """Append a dedicated practical HPO result row."""
        self._require_campaign("hpo")
        row = self._serialize_json_fields(
            row,
            ("optimizer_config", "best_params", "validation_trace"),
        )
        self._append_row(self.hpo_path, HPO_COLUMNS, row)

    @staticmethod
    def _serialize_json_fields(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        serialized = dict(row)
        for field in fields:
            value = serialized.get(field)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if value is not None and not isinstance(value, str):
                serialized[field] = json.dumps(
                    value,
                    default=ResultStore._json_default,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        return serialized

    def _require_campaign(self, run_kind: str) -> None:
        if not self._campaign_prepared:
            raise RuntimeError("prepare_campaign() must succeed before appending results")
        if self._run_kind != run_kind:
            raise RuntimeError(f"cannot append {run_kind} results to a {self._run_kind} campaign")

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert scalar/array-like scientific values without hiding invalid objects."""
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    def _append_row(self, path: Path, columns: list[str], row: dict[str, Any]) -> None:
        with self._lock:
            write_header = not path.exists()
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    def load_single(self) -> pd.DataFrame:
        """Load all single-objective results."""
        if not self.so_path.exists():
            return pd.DataFrame(columns=SO_COLUMNS)
        return pd.read_csv(self.so_path)

    def load_multi(self) -> pd.DataFrame:
        """Load all multi-objective results."""
        if not self.mo_path.exists():
            return pd.DataFrame(columns=MO_COLUMNS)
        return pd.read_csv(self.mo_path)

    def load_hpo(self) -> pd.DataFrame:
        """Load all dedicated practical HPO results."""
        if not self.hpo_path.exists():
            return pd.DataFrame(columns=HPO_COLUMNS)
        return pd.read_csv(self.hpo_path)

    def completed_so_runs(self) -> set[tuple[str, str, int, int]]:
        """Return set of (problem, optimizer, budget, run_id) already completed."""
        df = self.load_single()
        if df.empty:
            return set()
        return set(
            df[["problem", "optimizer", "budget", "run_id"]].itertuples(index=False, name=None)
        )

    def completed_mo_runs(self) -> set[tuple[str, str, int, int]]:
        """Return set of (problem, optimizer, budget, run_id) already completed."""
        df = self.load_multi()
        if df.empty:
            return set()
        return set(
            df[["problem", "optimizer", "budget", "run_id"]].itertuples(index=False, name=None)
        )

    def completed_hpo_runs(self) -> set[tuple[str, str, int, int]]:
        """Return completed dedicated HPO run keys."""
        df = self.load_hpo()
        if df.empty:
            return set()
        return set(
            df[["problem", "optimizer", "budget", "run_id"]].itertuples(index=False, name=None)
        )
