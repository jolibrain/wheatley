#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np


class CustomAgent:
    def __init__(self, env_cls, max_n_jobs, max_n_machines, rule="mopnr"):
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
        env = env_cls(
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
        observation, _ = env.reset()
        done = False
        while not done:
            action = self.select_action(observation)
            observation, _, done, _, _ = env.step(action)
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
