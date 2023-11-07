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
import math


class GraphTerminalRewardModel:
    def __init__(self, symlog_reward):
        self.symlog_reward = symlog_reward

    def evaluate(self, state):
        """
        Reward is 0 for every time steps, except for the last one, where it is the opposite of the Makespan
        """
        if state.succeeded():
            sinks = torch.where(state.types() == 1)[0]
            sinks_makespans = state.tct(sinks)
            max_makespan = torch.max(sinks_makespans)
            makespan = max_makespan.item()
            # makespan = state.tct(-1)[0].item() / len(state.job_modes)
            if self.symlog_reward:
                return -math.log1p(makespan)
            return -makespan / len(state.job_modes)
        if state.finished():
            if self.symlog_reward:
                return -math.log1p(state.undoable_makespan)
            return -state.undoable_makespan / len(state.job_modes)
        return 0
