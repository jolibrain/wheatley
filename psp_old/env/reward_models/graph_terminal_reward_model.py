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

import torch


class GraphTerminalRewardModel:
    def __init__(self):
        self.tardiness = False

    # def set_due_dates(self, due_dates):
    #     if due_dates is not None:
    #         with_due_dates = [i for i, v in enumerate(due_dates) if v != None]
    #         self.with_due_dates = torch.tensor(with_due_dates, dtype=torch.int64)
    #         self.due_dates = torch.tensor(
    #             [v for i, v in enumerate(due_dates) if v != None],
    #             dtype=torch.int64,
    #         )
    #     else:
    #         self.due_dates = None
    def set_tardiness(self):
        self.tardiness = True

    def evaluate(self, state):
        """
        Reward is 0 for every time steps, except for the last one, where it is the opposite of the Makespan
        """
        if state.succeeded():
            if not self.tardiness:
                sinks = torch.where(state.types() == 1)[0]
                sinks_makespans = state.tct_real(sinks)
                max_makespan = torch.max(sinks_makespans)
                makespan = max_makespan.item()
                # makespan = state.tct(-1)[0].item() / len(state.job_modes)
                return -makespan / len(state.job_modes)
            else:
                # wdd_tct = state.tct(self.with_due_dates)
                # tardy = wdd_tct - self.due_dates.unsqueeze(-1)
                raw_tardy = state.all_tardiness()
                tardy = torch.where(raw_tardy < 0, 0, raw_tardy)
                # num_has_due_date = state.graph.ndata("has_due_date").sum().item()
                # print("num_has_due_date", num_has_due_date)
                return (
                    -torch.sum(tardy).item()
                    / 3.0
                    # / num_has_due_date
                    / len(state.job_modes)
                    # / state.max_duration
                    # / num_has_due_date
                )  # average min,max,ave and come back close to 1

        if state.finished():
            if self.due_dates is not None:
                raise NotImplementedError("failure and tardiness")
            else:
                return -state.undoable_makespan / len(state.job_modes)
        return 0
