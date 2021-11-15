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

    solution = solve_jssp(affectations, durs, max_time_ortools, scaling_constant_ortools)

    if durations.shape[2] == 1:
        return solution.get_makespan(), solution.schedule

    state = State(affectations, durations, affectations.shape[0], affectations.shape[1])
    state.reset()

    # observe all real durations

    for i in range(n_j * n_m):
        state.observe_real_duration(i, do_update=True)
        state.affect_node(i)
    # we will use get_machine_occupancy_max_endtime which rely on task_completion_times
    state.task_completion_times = np.expand_dims(solution.schedule, 2) + durations

    occupancies = []

    for m in range(n_m):
        occupancies.append(state.get_machine_occupancy(m, ortools_strategy))

    state.reset()  # reset task_completion times in particular
    for i in range(n_j * n_m):
        state.observe_real_duration(i, do_update=True)
        state.affect_node(i)

    for o in occupancies:
        nodes_occ = [el[2] for el in o]
        for i in range(len(nodes_occ) - 1):
            state.set_precedency(nodes_occ[i], nodes_occ[i + 1])

    makespan = np.max(state.task_completion_times[:, :, 0])
    return makespan, state.task_completion_times[:, :, 0] - durations[:, :, 0]
