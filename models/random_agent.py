import numpy as np

from env.env import Env

from config import MAX_N_NODES


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
        real_mask = np.zeros((MAX_N_NODES, MAX_N_NODES))
        n_nodes = observation["n_nodes"]
        lil_mask = observation["mask"][0 : n_nodes * n_nodes].reshape(n_nodes, n_nodes)
        real_mask[0:n_nodes, 0:n_nodes] = lil_mask
        real_mask = real_mask.flatten()
        possible_actions = np.nonzero(real_mask)[0]
        action = np.random.choice(possible_actions)
        return action
