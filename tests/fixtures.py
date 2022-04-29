from gym.spaces import Dict, Box, Discrete
import numpy as np
from pytest import fixture
import torch
from torch_geometric.data import Data

from env.env import Env
from env.env_specification import EnvSpecification
from env.transition_models.l2d_transition_model import L2DTransitionModel
from env.state import State
from models.features_extractor import FeaturesExtractor
from problem.problem_description import ProblemDescription
from utils.env_observation import EnvObservation
from utils.agent_observation import AgentObservation


@fixture
def graph():
    graph = Data(
        x=torch.rand(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int64),
    )
    return graph


@fixture
def mask():
    mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return mask


@fixture
def problem_description():
    problem_description = ProblemDescription(
        transition_model_config="L2D",
        reward_model_config="L2D",
        deterministic=True,
        fixed=True,
        n_jobs=5,
        n_machines=5,
        max_duration=99,
    )
    return problem_description


@fixture
def env_specification():
    env_specification = EnvSpecification(
        max_n_jobs=5,
        max_n_machines=5,
        normalize_input=True,
        input_list=["duration"],
        insertion_mode="no_forced_insertion",
    )
    return env_specification


@fixture
def env(problem_description, env_specification):
    env = Env(problem_description, env_specification)
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
            [[1, 1, 1, 1], [5, 5, 5, 5], [10, 10, 10, 10], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[5, 5, 5, 5], [6, 6, 6, 6], [3, 3, 3, 3], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
            [[9, 9, 9, 9], [8, 8, 8, 8], [7, 7, 7, 7], [6, 6, 6, 6], [5, 5, 5, 5]],
        ]
    )
    return durations


@fixture
def state(affectations, durations):
    state = State(affectations, durations, 5, 5)
    return state


@fixture
def env_observation():
    return EnvObservation(
        n_jobs=3,
        n_machines=3,
        features=torch.tensor(
            [
                [0, 0, 0, 0, 15, 15, 15, 15],
                [1, 1, 1, 1, 2, 2, 2, 2],
                [1, 1, 1, 1, 5, 5, 5, 5],
                [1, 1, 1, 1, 7, 7, 7, 7],
                [0, 0, 0, 0, 15, 15, 15, 15],
                [0, 0, 0, 0, 14, 14, 14, 14],
                [1, 1, 1, 1, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 5, 5, 5, 5],
            ],
        ),
        edge_index=torch.tensor(
            [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            dtype=torch.int64,
        ),
        max_n_jobs=3,
        max_n_machines=3,
    )


@fixture
def agent_observation():
    mask = torch.zeros((2, 9))
    mask[:, 0] = 1
    mask[:, 3] = 1
    mask[:, 6] = 1
    return AgentObservation(
        n_jobs=3,
        n_machines=3,
        n_nodes=9,
        n_edges=8,
        features=torch.tensor(
            [
                [
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 5, 5, 5, 5],
                    [1, 1, 1, 1, 7, 7, 7, 7],
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [0, 0, 0, 0, 14, 14, 14, 14],
                    [1, 1, 1, 1, 4, 4, 4, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 5, 5, 5, 5],
                ],
                [
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 5, 5, 5, 5],
                    [1, 1, 1, 1, 7, 7, 7, 7],
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [0, 0, 0, 0, 14, 14, 14, 14],
                    [1, 1, 1, 1, 4, 4, 4, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 5, 5, 5, 5],
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
    mask = torch.zeros((2, 9), device=torch.device("cpu"))
    mask[:, 0] = 1
    mask[:, 3] = 1
    mask[:, 6] = 1
    return {
        "n_jobs": 3,
        "n_machines": 3,
        "n_nodes": 9,
        "n_edges": 8,
        "features": torch.tensor(
            [
                [
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 5, 5, 5, 5],
                    [1, 1, 1, 1, 7, 7, 7, 7],
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [0, 0, 0, 0, 14, 14, 14, 14],
                    [1, 1, 1, 1, 40, 40, 40, 40],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 5, 5, 5, 5],
                ],
                [
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 5, 5, 5, 5],
                    [1, 1, 1, 1, 7, 7, 7, 7],
                    [0, 0, 0, 0, 15, 15, 15, 15],
                    [0, 0, 0, 0, 14, 14, 14, 14],
                    [1, 1, 1, 1, 40, 40, 40, 40],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 5, 5, 5, 5],
                ],
            ],
            device=torch.device("cpu"),
        ),
        "edge_index": torch.tensor(
            [
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
                [[0, 2, 4, 5, 2, 3, 6, 7], [1, 3, 4, 2, 4, 5, 8, 6]],
            ],
            dtype=torch.int64,
            device=torch.device("cpu"),
        ),
        "mask": mask,
    }


@fixture
def l2d_transition_model(affectations, durations):
    l2d_transition_model = L2DTransitionModel(affectations, durations, 5, 5)
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
                "features": Box(0, 1, shape=(9, 8)),
                "edge_index": Box(0, 9, shape=(2, 81), dtype=np.int64),
                "mask": Box(0, 1, shape=(9,), dtype=np.int64),
            }
        ),
        input_dim_features_extractor=8,
        gconv_type="gin",
        freeze_graph=False,
        graph_pooling="max",
        graph_has_relu=False,
        device=torch.device("cpu"),
        max_n_nodes=25,
        n_mlp_layers_features_extractor=4,
        n_layers_features_extractor=4,
        hidden_dim_features_extractor=64,
        n_attention_heads=4,
    )
    return features_extractor
