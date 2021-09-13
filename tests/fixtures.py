from gym.spaces import Dict, Box, Discrete
import numpy as np
from pytest import fixture
import torch
from torch_geometric.data import Data

from env.env import Env
from env.l2d_transition_model import L2DTransitionModel
from env.state import State
from models.features_extractor import FeaturesExtractor
from problem.problem_description import ProblemDescription
from utils.env_observation import EnvObservation
from utils.agent_observation import AgentObservation

from config import DEVICE


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
def env_observation():
    mask = torch.zeros(81)
    mask[0] = 1
    mask[30] = 1
    mask[60] = 1
    return EnvObservation(
        n_jobs=3,
        n_machines=3,
        n_nodes=9,
        n_edges=8,
        features=torch.tensor(
            [
                [0, 15],
                [1, 2],
                [1, 5],
                [1, 7],
                [0, 15],
                [0, 14],
                [1, 4],
                [1, 1],
                [1, 5],
            ],
        ),
        edge_index=torch.tensor(
            [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            dtype=torch.int64,
        ),
        mask=mask,
    )


@fixture
def agent_observation():
    mask = torch.zeros((2, 81))
    mask[:, 0] = 1
    mask[:, 30] = 1
    mask[:, 60] = 1
    return AgentObservation(
        n_jobs=3,
        n_machines=3,
        n_nodes=9,
        n_edges=8,
        features=torch.tensor(
            [
                [
                    [0, 15],
                    [1, 2],
                    [1, 5],
                    [1, 7],
                    [0, 15],
                    [0, 14],
                    [1, 4],
                    [1, 1],
                    [1, 5],
                ],
                [
                    [0, 15],
                    [1, 2],
                    [1, 5],
                    [1, 7],
                    [0, 15],
                    [0, 14],
                    [1, 4],
                    [1, 1],
                    [1, 5],
                ],
            ],
        ),
        edge_index=torch.tensor(
            [
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            ],
            dtype=torch.int64,
        ),
        mask=mask,
    )


@fixture
def gym_observation():
    mask = torch.zeros((2, 81), device=DEVICE)
    mask[:, 0] = 1
    mask[:, 30] = 1
    mask[:, 60] = 1
    return {
        "n_jobs": 3,
        "n_machines": 3,
        "n_nodes": 9,
        "n_edges": 8,
        "features": torch.tensor(
            [
                [
                    [0, 15],
                    [1, 2],
                    [1, 5],
                    [1, 7],
                    [0, 15],
                    [0, 14],
                    [1, 40],
                    [1, 1],
                    [1, 5],
                ],
                [
                    [0, 15],
                    [1, 2],
                    [1, 5],
                    [1, 7],
                    [0, 15],
                    [0, 14],
                    [1, 40],
                    [1, 1],
                    [1, 5],
                ],
            ],
            device=DEVICE,
        ),
        "edge_index": torch.tensor(
            [
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            ],
            dtype=torch.int64,
            device=DEVICE,
        ),
        "mask": mask,
    }


@fixture
def l2d_transition_model(affectations, durations):
    l2d_transition_model = L2DTransitionModel(affectations, durations)
    return l2d_transition_model


@fixture
def features_extractor():
    features_extractor = FeaturesExtractor(
        observation_space=Dict(
            {
                "n_jobs": Discrete(3),
                "n_machines": Discrete(3),
                "n_nodes": Discrete(9),
                "n_edges": Discrete(81),
                "features": Box(0, 1, shape=(9, 2)),
                "edge_index": Box(0, 9, shape=(2, 81), dtype=np.int64),
                "mask": Box(0, 1, shape=(81,), dtype=np.int64),
            }
        )
    )
    return features_extractor
