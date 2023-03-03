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

from copy import deepcopy
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import visdom

from env.env import Env
from env.state import State
from models.random_agent import RandomAgent
from problem.problem_description import ProblemDescription
from utils.ortools_solver import solve_jssp
from utils.utils import generate_data
import torch


def get_ortools_makespan(
    affectations,
    durations,
    max_time_ortools,
    scaling_constant_ortools,
    ortools_strategy="pessimistic",
):
    n_j, n_m = affectations.shape[0], affectations.shape[1]

    if ortools_strategy == "realistic" or durations.shape[2] == 1:
        durs = durations[:, :, 0]
    elif ortools_strategy == "pessimistic":
        durs = durations[:, :, 2]
    elif ortools_strategy == "optimistic":
        durs = durations[:, :, 1]
    elif ortools_strategy == "averagistic":
        durs = durations[:, :, 3]
    else:
        print("unknow ortools strategy ", ortools_strategy)
        exit()

    solution = solve_jssp(affectations, durs, max_time_ortools, scaling_constant_ortools)

    if durations.shape[2] == 1:
        return solution.get_makespan(), solution.schedule

    state = State(affectations, durations, affectations.shape[0], affectations.shape[1])
    state.reset()

    # use the same durations to compute machines occupancies
    state.durations = state.original_durations
    for i in range(n_j * n_m):
        state.affect_node(i)
    # we will use get_machine_occupancy_max_endtime which rely on task_completion_times
    state.set_all_task_completion_times(np.expand_dims(solution.schedule, 2) + durations)

    occupancies = [state.get_machine_occupancy(m, ortools_strategy) for m in range(n_m)]

    state.reset()  # reset task_completion times in particular

    # observe all real durations
    for i in range(n_j * n_m):
        state.observe_real_duration(i, do_update=False)
        state.affect_node(i)

    for o in occupancies:
        nodes_occ = [el[2] for el in o]
        for i in range(len(nodes_occ) - 1):
            state.set_precedency(nodes_occ[i], nodes_occ[i + 1], do_update=False)

    state.update_completion_times_from_sinks()

    tct = state.get_all_task_completion_times()[:, 0].reshape(n_j, n_m, 1).squeeze_(2).numpy()

    makespan = torch.max(state.get_all_task_completion_times()[:, 0].flatten())

    return makespan, tct - durations[:, :, 0]
