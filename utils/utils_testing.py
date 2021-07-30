from copy import deepcopy
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import visdom

from env.env import Env
from models.random_agent import RandomAgent
from problem.problem_description import ProblemDescription
from utils.ortools_solver import solve_jssp
from utils.utils import generate_problem, generate_data, load_benchmark

from config import MAX_DURATION


def test_agent(agent, problem_description):
    solution = agent.predict(problem_description)
    makespan = np.max(solution.schedule + problem_description.durations)
    return makespan


def get_ortools_makespan(n_j, n_m, max_duration, affectations=None, durations=None):
    if affectations is None and durations is None:
        affectations, durations = generate_problem(n_j, n_m, max_duration)
    solution = solve_jssp(affectations, durations)
    makespan = np.max(solution.schedule + durations)
    return makespan
