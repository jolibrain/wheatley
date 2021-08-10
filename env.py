import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import torch

from config import MAX_N_NODES, MAX_N_EDGES


class Env(gym.Env):
    def __init__(self, n_jobs, n_machines, n_features):
        self.action_space = Discrete(MAX_N_EDGES)
        self.observation_space = Dict(
            {
                "features": Box(low=0, high=1, shape=(MAX_N_NODES, n_features)),
                "edge_index": Box(
                    low=0, high=MAX_N_NODES, shape=(2, MAX_N_EDGES), dtype=np.int64
                ),
                "n_nodes": Discrete(MAX_N_NODES),
            }
        )
        self.n_features = n_features
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_nodes = n_machines * n_jobs

    def step(self, action):
        observation = {}
        observation["n_nodes"] = self.n_nodes
        observation["features"] = np.zeros((MAX_N_NODES, self.n_features))
        observation["features"][0 : self.n_nodes] = np.random.rand(
            self.n_nodes, self.n_features
        )
        observation["edge_index"] = np.random.randint(
            0, self.n_nodes, size=(2, MAX_N_EDGES)
        )
        return observation, 0, False, {}

    def reset(self):
        o, _, _, _ = self.step(None)
        return o
