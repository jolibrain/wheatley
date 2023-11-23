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
import numpy as np


class TransitionModel:
    def __init__(
        self,
        env_specification,
    ):
        self.env_specification = env_specification

    def run(self, state, node_id):  # noqa
        state.affect_job(node_id)
        if self.env_specification.fast_forward:
            while True:
                unmasked_action = state.unmasked_actions()
                if unmasked_action.shape[0] == 1:
                    print("fast forwarding only action", unmasked_action[0])
                    state.affect_job(unmasked_action[0])
                    continue
                trivial_actions = state.trivial_actions()
                if trivial_actions.shape[0] > 0:
                    print("fast trivial action", trivial_actions[0])
                    state.affect_job(trivial_actions[0])
                    continue
                break

    def get_mask(self, state):
        return state.selectables() == 1
