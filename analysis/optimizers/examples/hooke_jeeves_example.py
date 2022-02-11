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
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination

from benchmarks.hard_to_optimize_functions.holder_table_2d import holder_table_np

# algorithm = PatternSearch()
termination = MaximumFunctionCallTermination(100)

algorithm = GA(pop_size=5, eliminate_duplicates=True)

problem = FunctionalProblem(n_var=2, objs=[holder_table_np], xl=-10, xu=10)


res = minimize(problem, algorithm, termination, verbose=True)
print(f"Best solution found: \nX = {res.X}\nF = {res.F}")
