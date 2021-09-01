import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import torch

from env.l2d_transition_model import L2DTransitionModel
from env.l2d_reward_model import L2DRewardModel
from utils.observation import Observation
from utils.utils import generate_problem

from config import MAX_N_NODES, MAX_N_EDGES, MAX_N_MACHINES


class Env(gym.Env):
    def __init__(self, problem_description):
        n_features = 3  # This is fixed by the way we choose the nodes features
        self.n_jobs = problem_description.n_jobs
        self.n_machines = problem_description.n_machines
        self.max_duration = problem_description.max_duration

        self.action_space = Discrete(MAX_N_EDGES)
        self.observation_space = Dict(
            {
                "features": Box(
                    # high is 99*n_machines due to lower bound method of calculation
                    low=0,
                    high=self.max_duration * MAX_N_MACHINES,
                    shape=(MAX_N_NODES, n_features),
                ),
                "edge_index": Box(
                    low=0,
                    high=MAX_N_NODES,
                    shape=(2, MAX_N_EDGES),
                    dtype=np.int64,
                ),
                "n_nodes": Discrete(MAX_N_NODES),
                "mask": Box(low=0, high=1, shape=(MAX_N_EDGES,)),
            }
        )
        self.n_nodes = self.n_machines * self.n_jobs

        self.affectations = problem_description.affectations
        self.durations = problem_description.durations
        self.transition_model_config = (
            problem_description.transition_model_config
        )
        self.reward_model_config = problem_description.reward_model_config

        self.generate_random_problems = False
        if (
            problem_description.affectations is None
            and problem_description.durations is None
        ):
            self.generate_random_problems = True

        self.transition_model = None
        self.reward_model = None

        self.reset()

    def step(self, action):
        print(action)
        obs = Observation.from_torch_geometric(
            self.transition_model.get_graph(), self.transition_model.get_mask()
        )
        self.transition_model.run(action)
        next_obs = Observation.from_torch_geometric(
            self.transition_model.get_graph(), self.transition_model.get_mask()
        )
        reward = self.reward_model.evaluate(obs, action, next_obs)
        done = self.transition_model.done()

        gym_observation = next_obs.to_gym_observation()

        return gym_observation, reward, done, {}

    def reset(self):
        if self.generate_random_problems:
            self.affectations, self.durations = generate_problem(
                self.n_jobs, self.n_machines, self.max_duration
            )
        self._create_transition_and_reward_model()
        observation = Observation.from_torch_geometric(
            self.transition_model.get_graph(), self.transition_model.get_mask()
        )
        return observation.to_gym_observation()

    def get_solution(self):
        return self.transition_model.state.get_solution()

    def _create_transition_and_reward_model(self):

        if self.transition_model_config == "L2D":
            self.transition_model = L2DTransitionModel(
                self.affectations, self.durations
            )
        else:
            raise Exception("Transition model not recognized")

        if self.reward_model_config == "L2D":
            self.reward_model = L2DRewardModel()
        else:
            raise Exception("Reward model not recognized")
