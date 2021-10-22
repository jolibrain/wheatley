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
from utils.utils import generate_problem, generate_data, load_benchmark

from config import MAX_DURATION


def test_agent(agent, problem_description, normalize_input, full_force_insert):
    solution = agent.predict(problem_description, normalize_input, full_force_insert)
    makespan = np.max(solution.schedule + problem_description.durations)
    return makespan


def get_ortools_makespan(n_j, n_m, max_duration, affectations=None, durations=None, ortools_strategy='pessimistic'):
    if affectations is None and durations is None:
        affectations, durations = generate_problem(n_j, n_m, max_duration)

    if ortools_strategy == "realistic" or durations.shape[2] == 1:
        durs = durations[:,:,0]
    elif ortools_strategy == "pessimistic":
        durs = durations[:,:,2]
    elif ortools_strategy == "optimistic":
        durs = durations[:,:,1]
    elif ortools_strategy == "averagistic":
        durs = durations[:,:,3]

    solution = solve_jssp(affectations, durs)

    if durations.shape[2] == 1:
        makespan = np.max(solution.schedule + durations[:,:,0])
        return makespan, solution.schedule

    state = State(affectations, durations)
    state.reset()

    #observe all real durations

    for i in range(n_j * n_m):
        state.observe_real_duration(i, do_update=True)
        state.affect_node(i)
    # we will use get_machine_occupancy_max_endtime which rely on task_completion_times
    state.task_completion_times = np.expand_dims(solution.schedule,2) + durations

    occupancies = []

    for m in range(n_m):
        occupancies.append(state.get_machine_occupancy(m))

    state.reset() # reset task_completion times in particular
    for i in range(n_j * n_m):
        state.observe_real_duration(i, do_update=True)
        state.affect_node(i)

    for o in occupancies:
        nodes_occ = [el[2] for el in o]
        for i in range(len(nodes_occ)-1):
            state.set_precedency(nodes_occ[i], nodes_occ[i+1])

    makespan = np.max(state.task_completion_times[:,:,0])
    return makespan, state.task_completion_times[:,:,0] - durations[:,:,0]
