import numpy as np
import torch

from env.env import Env


class RandomAgent:
    def __init__(self):
        pass

    def predict(self, problem_description):
        env = Env(problem_description)
        observation = env.reset()
        done = False
        while not done:
            action = self.select_action(observation)
            observation, _, done, _ = env.step(action)
        solution = env.get_solution()
        return solution

    def select_action(self, observation):
        possible_actions = (
            torch.nonzero(observation["mask"], as_tuple=True)[0]
            .detach()
            .numpy()
        )
        return np.random.choice(possible_actions)
