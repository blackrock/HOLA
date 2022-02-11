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

from logging import getLogger
from multiprocessing import Pool
from typing import Callable, Iterable, Sequence

import pandera.typing as pat
from pandas import Timestamp, concat
from returns.curry import partial

from analysis.infra.data_persister import Analysis, ResearchDataset

logger = getLogger()


def compare_optimizers(
    benchmark_optimizers: Sequence[Callable[[int], pat.DataFrame[Analysis]]],
    research_dataset: ResearchDataset,
    n_repetitions: int = 20,
    iterations: Iterable[int] = (300, 500, 1000),
) -> None:
    for max_iterations in iterations:
        res: list[pat.DataFrame[Analysis]] = []
        print("=" * 20, f"MAX_ITERATIONS={max_iterations}", "=" * 20)
        for i in range(1, n_repetitions + 1):
            intermediate_res: list[pat.DataFrame[Analysis]] = []
            for runner in benchmark_optimizers:
                df = runner(max_iterations)
                res.append(df.copy())
                intermediate_res.append(df)
            print(f"\n{i}/{n_repetitions}\n{concat(intermediate_res).sort_values(Analysis.best)}")
        research_dataset.write_to_dataset(concat(res))  # type: ignore[arg-type]


def run_parallel_by_optimizer(
    benchmark_optimizers: list[Callable[[int], pat.DataFrame[Analysis]]],
    research_dataset: ResearchDataset,
    n_processes: int = 5,
    n_repetitions: int = 20,
    max_evaluations: tuple[int, ...] = (300, 500, 1000),
) -> None:
    """Parallelizes per optimizer."""
    with Pool(n_processes) as pool:
        pool.map(
            partial(
                repeat_optimization,
                max_evaluations=max_evaluations,
                n_repetitions=n_repetitions,
                research_dataset=research_dataset,
            ),
            benchmark_optimizers,
        )


def repeat_optimization(
    func: Callable[[int], pat.DataFrame[Analysis]],
    research_dataset: ResearchDataset,
    max_evaluations: Iterable[int],
    n_repetitions: int,
) -> None:
    for evaluations in max_evaluations:
        dfs: list[pat.DataFrame[Analysis]] = []
        for _ in range(n_repetitions):
            start = Timestamp.now()
            print(f"{start}: Starting {func} with {evaluations}")  # Not worth setting up the logger in the process
            res = func(evaluations)
            end = Timestamp.now()
            print(f"{end}: Finished {func} with {evaluations}. Took: {end - start}")
            dfs.append(res)
        df = concat(dfs)
        research_dataset.write_to_dataset(df)  # type: ignore[arg-type]
        logger.info(f"Wrote results of {n_repetitions} repetitions of {func} ran with {evaluations} iterations to disk")


def run_full_parallel(
    benchmark_optimizers: list[Callable[[int], pat.DataFrame[Analysis]]],
    research_dataset: ResearchDataset,
    n_processes: int = 5,
    n_repetitions: int = 20,
    max_evaluations: tuple[int, ...] = (300, 500, 1000),
) -> None:
    """Parallelizes per optimizer, number of evaluations and repetition, but does not yield intermediate results."""
    with Pool(n_processes) as pool:
        logger.info("Set up process pool")
        jobs = [
            (optimizer_runner, evaluations)
            for optimizer_runner in benchmark_optimizers
            for evaluations in max_evaluations
            for _ in range(n_repetitions)
        ]
        logger.info(f"Set up {len(jobs)} jobs: {jobs}")
        res = pool.map(repeat_optimization_2, jobs)
        # No intermediate results
        research_dataset.write_to_dataset(concat(res))  # type: ignore[arg-type]


def repeat_optimization_2(
    func_to_evaluations: tuple[Callable[[int], pat.DataFrame[Analysis]], int],
) -> pat.DataFrame[Analysis]:
    func, evaluations = func_to_evaluations
    start = Timestamp.now()
    print(f"{start}: Starting {func} with {evaluations}")  # Not worth setting up the logger in the process
    res = func(evaluations)
    end = Timestamp.now()
    print(f"{end}: Finished {func} with {evaluations}. Took: {end - start}")
    return res


def run_sequentially(
    benchmark_optimizers: Sequence[Callable[[int], pat.DataFrame[Analysis]]],
    research_dataset: ResearchDataset,
    n_repetitions: int = 20,
    max_evaluations: tuple[int, ...] = (300, 500, 1000),
) -> None:
    for func in benchmark_optimizers:
        for evaluations in max_evaluations:
            res: list[pat.DataFrame[Analysis]] = []
            for _ in range(1, n_repetitions + 1):
                start = Timestamp.now()
                print(f"{start}: Starting {func} with {evaluations}")
                df = func(evaluations)
                end = Timestamp.now()
                print(f"{end}: Finished {func} with {evaluations}. Took: {end - start}")
                res.append(df.copy())
            research_dataset.write_to_dataset(concat(res))  # type: ignore[arg-type]
