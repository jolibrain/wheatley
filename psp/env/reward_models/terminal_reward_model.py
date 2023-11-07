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
import math


class TerminalRewardModel:
    def __init__(self, symlog_reward):
        self.symlog_reward = symlog_reward

    def evaluate(self, state):
        """
        Reward is 0 for every time steps, except for the last one, where it is the opposite of the Makespan
        """
        if state.succeeded():
            sinks = np.where(state.types() == 1)[0]
            sinks_makespans = state.tct(sinks)
            max_makespan = np.max(sinks_makespans)
            if self.symlog_reward:
                # return -math.log1p(max_makespan)
                return -max_makespan  # / len(state.job_modes)
            return -max_makespan / len(state.job_modes)
        if state.finished():
            if self.symlog_reward:
                # return -math.log1p(state.undoable_makespan)
                return -max_makespan  # / len(state.job_modes)
            return -state.undoable_makespan / len(state.job_modes)
        return 0
