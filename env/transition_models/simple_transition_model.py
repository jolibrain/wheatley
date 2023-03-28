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


class SimpleTransitionModel(TransitionModel):
    def __init__(
        self,
        affectations,
        durations,
        max_n_jobs,
        max_n_machines,
        observe_real_duration_when_affect=True,
        stochastic_metric="pessimistic",
    ):
        super(SimpleTransitionModel, self).__init__(
            affectations, durations, max_n_jobs, max_n_machines
        )
        self.useless_timesteps = 0
        self.observe_real_duration_when_affect = observe_real_duration_when_affect
        self.metric = stochastic_metric
        if self.metric == "realistic":
            self.metric_index = 0
        elif self.metric == "optimistic":
            self.metric_index = 1
        elif self.metric == "pessimistic":
            self.metric_index = 2
        elif self.metric == "averagistic":
            self.metric_index = 3
        else:
            raise Exception("Stochastic metric not recognized")

    def run(self, state, node_id, force_insert):  # noqa

        # If the job_id is bigger that max job_id, we don't operate the action
        job_id, task_id = node_to_job_and_task(node_id, state.n_machines)
        if job_id >= state.n_jobs:
            self.useless_timesteps += 1
            return

        if task_id != state.get_first_unaffected_task(job_id):
            self.useless_timesteps += 1
            return

        machine_id = state.affectations[job_id, task_id]
        if machine_id == -1:
            self.useless_timesteps += 1
            return

        # machine_occupancy = state.get_machine_occupancy(machine_id, self.metric)
        last_task_on_machine = state.get_last_task_on_machine(machine_id, self.metric)

        # Observe duration (for the uncertainty case)
        state.observe_real_duration(
            node_id,
            do_update=False,
            update_duration_with_real=self.observe_real_duration_when_affect,
        )

        if last_task_on_machine is not None:
            state.set_precedency(last_task_on_machine, node_id, do_update=False)
        state.cache_last_task_on_machine(machine_id, node_id)
        state.update_completion_times(node_id)
        state.affect_node(node_id)

    def get_mask(self, state, add_boolean=False):
        # mask = np.full((state.n_nodes,), False, dtype=bool)
        # for job_id in range(state.n_jobs):
        #     task_id = state.get_first_unaffected_task(job_id)
        #     if task_id != -1 and state.affectations[job_id, task_id] != -1:
        #         mask[job_and_task_to_node(job_id, task_id, state.n_machines)] = True
        # return mask
        return state.get_selectable() == 1
