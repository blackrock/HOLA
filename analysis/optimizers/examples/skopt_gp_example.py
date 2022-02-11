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
# from __future__ import annotations
#
# from typing import Iterable
#
# from skopt import forest_minimize
#
# from benchmarks.hard_to_optimize_functions.rastrigin_nd import rastrigin
#
# PROGRESS = 0
#
#
# def progress_rastrigin(x: float | Iterable[float]) -> float:
#     global PROGRESS  # pylint: disable=global-statement
#     PROGRESS += 1
#     print("Rastrigin progress", PROGRESS)
#     return rastrigin(x)
#
#
# res = forest_minimize(  # Very slow
#     progress_rastrigin,  # the function to minimize
#     [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)],  # the bounds on each dimension of x
#     # acq_func="gp_hedge",  # the acquisition function
#     # Other acquisition functions:
#     # - "LCB"` for lower confidence bound.
#     # - `"EI"` for negative expected improvement.
#     # - `"PI"` for negative probability of improvement.
#     # - `"gp_hedge"` Probabilistically choose one of the above three
#     n_calls=300,  # the number of evaluations of f
#     # n_random_starts=20,  # the number of random initialization points
# )
# print(rastrigin(res.x), res.x)
