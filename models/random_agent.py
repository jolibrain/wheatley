import numpy as np

from env.env import Env


class RandomAgent:
    def __init__(self, max_n_jobs, max_n_machines):
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_n_nodes = self.max_n_jobs * self.max_n_machines

    def predict(self, problem_description, env_specification):
        env = Env(
            problem_description,
            env_specification,
        )
        observation = env.reset()
        done = False
        while not done:
            action = self.select_action(observation)
            observation, _, done, _ = env.step(action)
        solution = env.get_solution()
        return solution

    def select_action(self, observation):
        real_mask = np.zeros(self.max_n_jobs)
        mask = observation["mask"]
        real_mask = mask.flatten()
        possible_actions = np.nonzero(real_mask)[0]
        action = np.random.choice(possible_actions)
        return action
