import gym
from gym.spaces import Discrete

from env.graph_space import GraphSpace

MAX_JOBS = 10
MAX_MACHINES = 10
MAX_ALLOWED_NUMBER_ACTIONS = (MAX_JOBS * MAX_MACHINES)**2

class Env(gym.Env):
    
    def __init__(self, n_jobs, n_machines, n_features):
        self.action_space = Discrete(MAX_ALLOWED_NUMBER_ACTIONS)
        self.observation_space = GraphSpace(n_features)
        self.n_jobs = n_jobs
        self.n_machines = n_machines

    def step(self, action):
        return self.observation_space.sample(), 0, False, None

    def reset(self):
        return self.observation_space.sample()
