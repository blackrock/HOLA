# Copyright 2021 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandera as pa
import pandera.typing as pat
from pandas import DataFrame, DatetimeTZDtype, Timestamp, set_option
from pandera import Field, SchemaModel
from pyarrow import Table
from pyarrow.parquet import ParquetDataset, write_to_dataset

set_option("display.width", 3000)
set_option("display.max_columns", 80)
set_option("display.max_colwidth", None)


class Analysis(SchemaModel):
    benchmark: pat.Series[str] = Field(coerce=True)
    optimizer: pat.Series[str] = Field(coerce=True)
    configuration: pat.Series[int] = Field(coerce=True)
    num_iterations: pat.Series[int] = Field(coerce=True)
    best: pat.Series[float]
    best_params: pat.Series[str] = Field(nullable=True)  # Stringified version

    @classmethod
    def build_row(
        cls, benchmark: str, optimizer: str, configuration: int, num_iterations: int, best: float, best_params: str
    ) -> pat.DataFrame[Analysis]:
        df = DataFrame(
            {
                Analysis.benchmark: [benchmark],
                Analysis.optimizer: [optimizer],
                Analysis.configuration: [configuration],
                Analysis.num_iterations: [num_iterations],
                Analysis.best: best,
                Analysis.best_params: best_params,
            }
        )
        return Analysis.validate(df)  # type: ignore[return-value]


class Analysis2(Analysis):
    time: pat.Series[DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC"})  # type: ignore[type-var]


@dataclass
class ResearchDataset:
    path: Path = Path(__file__).parents[1] / "data/pq_data_3"
    partition_cols: tuple[str, ...] = (
        Analysis.benchmark,
        Analysis.optimizer,
        Analysis.configuration,
        Analysis.num_iterations,
    )

    def write_to_dataset(self, df: pat.DataFrame[Analysis] | pat.DataFrame[Analysis2]) -> None:
        if Analysis2.time not in df.columns:
            df[Analysis2.time] = Timestamp.utcnow()
        write_to_dataset(
            Table.from_pandas(Analysis2.validate(df), preserve_index=False),
            root_path=self.path,
            partition_cols=list(self.partition_cols),
        )
        print(f"Written to: {self.path}")

    def read_data(self, dnf_filters: list[tuple] | list[list[tuple]] | None = None) -> pat.DataFrame[Analysis2]:
        df = ParquetDataset(self.path, filters=dnf_filters).read_pandas().to_pandas()
        return Analysis2.validate(df)  # type: ignore[return-value]

    @property
    def stats(self) -> DataFrame:
        return compute_stats(self.read_data())


def compute_stats(df: pat.DataFrame[Analysis2], col: str = Analysis2.best) -> DataFrame:
    group = df.groupby([Analysis.benchmark, Analysis.num_iterations, Analysis.optimizer, Analysis.configuration])[col]
    return DataFrame(
        {
            "mean": group.mean(),
            "std": group.std(),
            "min": group.min(),
            "counts": group.count(),
            "median": group.median(),
        }
    )


def normalize(
    data: pat.DataFrame[Analysis2],
    by: Iterable[str] | None = (Analysis2.benchmark, Analysis2.num_iterations),
    on: str = Analysis2.best,
) -> DataFrame:
    scored = data.copy()
    min_col, max_col = f"min_of_{on}", f"max_of_{on}"
    group = scored.groupby(list(by) if by is not None else None)[on]
    scored[min_col], scored[max_col] = group.transform("min"), group.transform("max")
    scored["score"] = (scored[on] - scored[min_col]) / (scored[max_col] - scored[min_col])
    return scored.drop(columns=[min_col, max_col])


def compute_stats_of_normalized(data: pat.DataFrame[Analysis2]) -> DataFrame:
    score_stats = compute_stats(normalize(data), col="score").reset_index()
    df = score_stats.pivot(index=["num_iterations", "optimizer", "configuration"], columns="benchmark", values="mean")
    df["mean_of_scores"] = df.mean(axis=1)
    df = df.sort_values(["num_iterations", "mean_of_scores"], ascending=[False, True])
    return df
