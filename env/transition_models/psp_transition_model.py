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

from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node, node_to_job_and_task


class PSPTransitionModel(TransitionModel):
    def __init__(
        self,
        env_specification,
        observe_real_duration_when_affect=False,
    ):
        self.observe_real_duration_when_affect = observe_real_duration_when_affect

    def run(self, state, node_id):  # noqa
        pass

    def get_mask(self, state, add_boolean=False):
        return state.get_selectable() == 1
