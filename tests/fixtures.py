import numpy as np
from pytest import fixture
import torch
from torch_geometric.data import Data

from env.env import Env
from env.state import State
from problem.problem_description import ProblemDescription


@fixture
def graph():
    graph = Data(
        x=torch.rand(4, 2),
        edge_index=torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int64
        ),
    )
    return graph


@fixture
def problem_description():
    problem_description = ProblemDescription(5, 5, 99, "L2D", "L2D")
    return problem_description


@fixture
def env(problem_description):
    env = Env(problem_description)
    return env


@fixture
def affectations():
    affectations = np.array(
        [
            [0, 2, 3, 1, 4],
            [2, 4, 3, 1, 0],
            [0, 1, 2, 3, 4],
            [4, 1, 0, 3, 2],
            [2, 3, 4, 1, 0],
        ]
    )
    return affectations


@fixture
def durations():
    durations = np.array(
        [
            [1, 5, 10, 7, 8],
            [5, 6, 3, 3, 4],
            [4, 4, 4, 4, 4],
            [5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5],
        ]
    )
    return durations


@fixture
def state(affectations, durations):
    state = State(affectations, durations)
    return state
