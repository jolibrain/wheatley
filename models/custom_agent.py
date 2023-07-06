#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

from dispatching_rules.solver import Solver
from problem.jssp_description import JSSPDescription
from problem.solution import Solution


class CustomAgent:
    def __init__(self, rule: str = "MOPNR"):
        self.rule = rule

    def predict(self, problem_description: JSSPDescription) -> Solution:
        machines = problem_description.affectations
        processing_times = problem_description.durations
        processing_times = processing_times.mean(
            axis=2
        )  # TODO: Handle the non-deterministic case.
        solver = Solver(
            processing_times, machines, self.rule, ignore_unfinished_precedences=True
        )
        schedule = solver.solve()
        solution = Solution(schedule, processing_times)
        return solution
