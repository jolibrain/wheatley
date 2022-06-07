import numpy as np

from env.env import Env
from sb3_contrib.common.maskable.utils import get_action_masks


class RandomAgent:
    def __init__(self, max_n_jobs, max_n_machines):
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_n_nodes = self.max_n_jobs * self.max_n_machines

    def predict(self, env):
        # soft reset to evaluate the same sampled problem as PPO
        observation = env.reset(soft=True)
        done = False
        while not done:
            action = self.select_action(env)
            observation, _, done, _ = env.step(action)
        solution = env.get_solution()
        return solution

    def select_action(self, env):
        action_masks = get_action_masks(env)
        possible_actions = np.nonzero(action_masks)[0]
        action = np.random.choice(possible_actions)
        return action
