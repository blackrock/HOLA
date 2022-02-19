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

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Iterable, List, MutableMapping, Sequence, Type, TypeVar, cast

from pandas import DataFrame

from hola.params import ParamConfig, ParamName, ParamsSpec, parse_params_config

TNum = TypeVar("TNum")
THypercube = TypeVar("THypercube")
Generation = int


@dataclass
class Side:
    """Coordinates of the sides of a hypercube (i.e. a dimension of the hypercube)."""

    lower: float
    upper: float

    @property
    def length(self) -> float:
        return self.upper - self.lower

    def get_lattices(self, spacing: int) -> list[float]:
        """Split the side of the hypercube in equal segments."""
        return [self.lower + self.length * i / spacing for i in range(spacing + 1)]

    def shrink_around(self, new_center: float, spacing: int) -> Side:
        new_lower = max(new_center - self.length / (2 * spacing), 0)
        new_upper = min(new_center + self.length / (2 * spacing), 1)
        return Side(new_lower, new_upper)


@dataclass
class Hypercube:
    """All sides are normalized (i.e. the Side-s are contained in the [0, 1] interval)."""

    sides: dict[ParamName, Side]

    @classmethod
    def unit(cls: Type[THypercube], params_config: dict[ParamName, ParamConfig]) -> THypercube:
        """Initial hypercube, all sides are [0, 1]."""
        return cls({param_name: Side(0, 1) for param_name in params_config})  # type: ignore[call-arg]  # false positive

    def get_lattices(self, spacing: int) -> list[dict[ParamName, float]]:
        """Split the hypercube in equally sized smaller sub-hypercubes, i.e. find lattices."""
        lattices_per_param = {param_name: side.get_lattices(spacing) for param_name, side in self.sides.items()}
        diagonal_lattices_as_df = DataFrame(lattices_per_param)
        params = list(self.sides)
        lattice_df = diagonal_lattices_as_df[[params[0]]]
        for param in params[1:]:
            lattice_df = lattice_df.merge(diagonal_lattices_as_df[[param]], how="cross")  # type: ignore[arg-type]
        return cast(List[Dict[ParamName, float]], lattice_df.to_dict(orient="records"))

    def shrink_around(self, new_center: dict[ParamName, float], spacing: int) -> Hypercube:
        return Hypercube(
            {
                param_name: self.sides[param_name].shrink_around(value, spacing)
                for param_name, value in new_center.items()
            }
        )


@dataclass(order=True)
class Evaluation(Generic[TNum]):
    """Results of evaluating the objective function along with its inputs."""

    params: dict[ParamName, float] = field(compare=False)
    val: TNum


class IterativeGridRefinement:
    def __init__(self, params_config: ParamsSpec, spacing: int):
        self.params_config = parse_params_config(params_config)
        assert all(config.param_type == "float" for config in self.params_config.values()), "IGR only supports float-s"
        self.num_params = len(self.params_config)
        if self.num_params >= 4:
            warnings.warn("IGR is not recommended for dimensionality >= 4")
        self.spacing = spacing

    def tune(
        self, func: Callable[[Iterable[float]], TNum] | Callable[[Sequence[float]], TNum], max_iterations: int
    ) -> Evaluation[TNum]:
        """Tune the given function, stopping after max_generations generations.

        A generation calls func many times.
        """
        hypercube = Hypercube.unit(self.params_config)
        max_generations = self.get_number_of_generations(max_iterations)
        assert self.get_number_of_iterations(max_generations) >= max_iterations
        iteration = 0
        params_tried: set[tuple[float, ...]] = set()
        evaluations: MutableMapping[Generation, list[Evaluation]] = defaultdict(list)
        for generation in range(max_generations):
            # Find lattices in the normalized hypercube
            norm_samples = hypercube.get_lattices(self.spacing)
            # Map params back to their domain and evaluate the function
            for sample in norm_samples:
                denorm_params = {
                    param_name: config.transform_param(sample[param_name])
                    for param_name, config in self.params_config.items()
                }
                params_to_try = tuple(denorm_params.values())
                if params_to_try not in params_tried:  # Check if we have already run this combination
                    res = func(params_to_try)
                    evaluations[generation].append(Evaluation(denorm_params, res))
                    params_tried.add(params_to_try)
                    iteration += 1
                    if iteration >= max_iterations:
                        return self.get_best_evaluation(evaluations)
            if not evaluations[generation]:
                break
            best_evaluation_in_generation = min(evaluations[generation])
            # Shrink hypercube around the normalized best params of this generation
            hypercube = hypercube.shrink_around(
                {
                    param_name: config.back_transform_param(best_evaluation_in_generation.params[param_name])
                    for param_name, config in self.params_config.items()
                },
                self.spacing,
            )
        return self.get_best_evaluation(evaluations)

    def get_number_of_iterations(self, max_generations: int) -> int:
        """Used to probe the number of iterations since IGR is defined in terms of generations."""
        return cast(int, (self.spacing + 1) ** self.num_params * max_generations)

    def get_number_of_generations(self, max_iterations: int) -> int:
        """Used to find the number of generations since IGR refines in generations."""
        res = max_iterations // ((self.spacing + 1) ** self.num_params)
        return cast(int, res + 1)

    @staticmethod
    def get_best_evaluation(evaluations: MutableMapping[Generation, list[Evaluation]]) -> Evaluation:
        return min(min(evals) for evals in evaluations.values() if evals)
