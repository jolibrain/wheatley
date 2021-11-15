import numpy as np

from env.env import Env


class CustomAgent:
    def __init__(self, max_n_jobs, max_n_machines, rule="mopnr"):
        self.index = None
        if rule == "mopnr":
            self.index = 0
        elif rule == "mwkr":
            self.index = 1
        elif rule == "cr":
            self.index = 2
        else:
            raise Exception("Rule not recognized")
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines

    def predict(self, problem_description, normalize_input, full_force_insert):
        env = Env(
            problem_description,
            self.max_n_jobs,
            self.max_n_machines,
            True,
            [
                "one_hot_machine_id",
                "one_hot_job_id",
                "duration",
                "total_job_time",
                "total_machine_time",
                "job_completion_percentage",
                "machine_completion_percentage",
                "mopnr",
                "mwkr",
                "cr",
            ],
        )
        observation = env.reset()
        done = False
        while not done:
            action = self.select_action(observation)
            observation, _, done, _ = env.step(action)
        solution = env.get_solution()
        return solution

    def select_action(self, observation):
        real_mask = np.zeros((self.max_n_nodes, self.max_n_nodes))
        n_nodes = observation["n_nodes"]
        lil_mask = observation["mask"][0 : n_nodes * n_nodes].reshape(n_nodes, n_nodes)
        real_mask[0:n_nodes, 0:n_nodes] = lil_mask
        features = observation["features"]
        for node_id, feat in enumerate(features):
            if node_id >= 2 * n_nodes:
                break
            real_mask[node_id, node_id] = real_mask[node_id, node_id] * (
                feat[self.max_n_jobs + self.max_n_machines + 7 + self.index] + 10
            )  # we add +10 to avoid using other actions than these one
        real_mask = real_mask.flatten()
        action = np.argmax(real_mask)
        return action
