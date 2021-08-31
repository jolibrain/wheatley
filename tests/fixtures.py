import numpy as np
from pytest import fixture
import torch
from torch_geometric.data import Data

from env.env import Env
from env.l2d_transition_model import L2DTransitionModel
from env.state import State
from problem.problem_description import ProblemDescription
from utils.observation import Observation


@fixture
def graph():
    graph = Data(
        x=torch.rand(4, 3),
        edge_index=torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int64
        ),
    )
    return graph


@fixture
def mask():
    mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return mask


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


@fixture
def observation():
    return Observation(
        n_nodes=9,
        features=torch.tensor(
            [
                [
                    [0, 0, 15],
                    [1, 1, 2],
                    [3, 1, 5],
                    [2, 1, 7],
                    [4, 0, 15],
                    [7, 0, 14],
                    [8, 1, 4],
                    [5, 1, 1],
                    [6, 1, 5],
                ],
            ]
        ),
        edge_index=torch.tensor(
            [
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            ],
            dtype=torch.int64,
        ),
        mask=torch.tensor([[1, 1, 1] + [0 for i in range(78)]]),
    )


@fixture
def batched_observation():
    return Observation(
        n_nodes=9,
        features=torch.tensor(
            [
                [
                    [0, 0, 15],
                    [1, 1, 2],
                    [3, 1, 5],
                    [2, 1, 7],
                    [4, 0, 15],
                    [7, 0, 14],
                    [8, 1, 4],
                    [5, 1, 1],
                    [6, 1, 5],
                ],
                [
                    [0, 0, 15],
                    [1, 1, 2],
                    [3, 1, 5],
                    [2, 1, 7],
                    [4, 0, 15],
                    [7, 0, 14],
                    [8, 1, 4],
                    [5, 1, 1],
                    [6, 1, 5],
                ],
            ]
        ),
        edge_index=torch.tensor(
            [
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            ],
            dtype=torch.int64,
        ),
        mask=torch.tensor(
            [
                [1, 1, 1] + [0 for i in range(78)],
                [1, 0, 1, 1] + [0 for i in range(77)],
            ]
        ),
    )


@fixture
def gym_observation():
    return {
        "n_nodes": 9,
        "features": torch.tensor(
            [
                [
                    [0, 0, 15],
                    [1, 1, 2],
                    [3, 1, 5],
                    [2, 1, 7],
                    [4, 0, 15],
                    [7, 0, 14],
                    [8, 1, 4],
                    [5, 1, 1],
                    [6, 1, 5],
                ],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            ],
            dtype=torch.int64,
        ),
        "mask": torch.tensor([[1, 1, 1] + [0 for i in range(78)]]),
    }


@fixture
def l2d_transition_model(affectations, durations):
    l2d_transition_model = L2DTransitionModel(affectations, durations)
    return l2d_transition_model
