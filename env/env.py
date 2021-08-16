import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import torch

from env.l2d_transition_model import L2DTransitionModel
from env.reward_model import RewardModel
from utils.state import State
from utils.utils import generate_problem

from config import MAX_N_NODES, MAX_N_EDGES


class Env(gym.Env):
    def __init__(self, n_jobs, n_machines):
        n_features = 2  # This is fixed by the way we choose the nodes features
        self.action_space = Discrete(MAX_N_EDGES)
        self.observation_space = Dict(
            {
                "features": Box(
                    # high is 99*n_machines due to lower bound method of calculation
                    low=0,
                    high=99 * n_machines,
                    shape=(MAX_N_NODES, n_features),
                ),
                "edge_index": Box(
                    low=0,
                    high=MAX_N_NODES,
                    shape=(2, MAX_N_EDGES),
                    dtype=np.int64,
                ),
                "n_nodes": Discrete(MAX_N_NODES),
            }
        )
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_nodes = n_machines * n_jobs

        self.affectations, self.durations = generate_problem(
            n_jobs, n_machines, low=0, high=99
        )

        self.transition_model = L2DTransitionModel(
            n_jobs, n_machines, self.affectations, self.durations
        )
        self.reward_model = RewardModel()

    def step(self, action):
        state = State.from_graph(self.transition_model.get_graph())
        self.transition_model.step(action)
        next_state = State.from_graph(self.transition_model.get_graph())
        reward = self.reward_model.evaluate(state, action, next_state)
        done = self.transition_model.done()

        observation = next_state.to_observation()

        return observation, reward, done, {}

    def reset(self):
        self.transition_model.reset()
        state = State.from_graph(self.transition_model.get_graph())
        return state.to_observation()
