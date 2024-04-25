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


class TerminalRewardModel:
    def __init__(self):
        pass

    def evaluate(self, state):
        """
        Reward is 0 for every time steps, except for the last one, where it is the opposite of the Makespan
        """
        if state.succeeded():
            sinks = np.where(state.types() == 1)[0]
            sinks_makespans = state.tct(sinks)
            max_makespan = np.max(sinks_makespans)
            return -max_makespan / len(state.job_modes)
            # BELOW JSSP reward
            # return -max_makespan / state.max_duration / 2
        if state.finished():
            return -state.undoable_makespan / len(state.job_modes)
            # BELOW JSSP reward
            # return -state.undoable_makespan / state.max_duration / 2
        return 0
