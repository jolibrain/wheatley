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

import numpy as np

from dispatching_rules.solver import Solver, reschedule
from problem.solution import Solution


class CustomAgent:
    def __init__(
        self, rule: str = "MOPNR", stochasticity_strategy: str = "averagistic"
    ):
        self.rule = rule
        self.stochasticity_strategy = stochasticity_strategy

    def predict(self, durations: np.ndarray, affectations: np.ndarray) -> Solution:
        real_durations = durations[:, :, 0]

        if self.stochasticity_strategy == "realistic" or durations.shape[2] == 1:
            durations = durations[:, :, 0]
        elif self.stochasticity_strategy == "pessimistic":
            durations = durations[:, :, 2]
        elif self.stochasticity_strategy == "optimistic":
            durations = durations[:, :, 1]
        elif self.stochasticity_strategy == "averagistic":
            durations = durations[:, :, 3]
        else:
            raise ValueError(
                f"Unknown stochasticity strategy {self.stochasticity_strategy}"
            )

        solver = Solver(
            durations, affectations, self.rule, ignore_unfinished_precedences=True
        )
        schedule = solver.solve()
        true_schedule = reschedule(real_durations, affectations, schedule)
        solution = Solution(true_schedule, durations)
        return solution
